# liveapi.py - Windows Compatible Version
import asyncio
import json
import logging
import os
import time
from pathlib import Path
import zmq
import threading
from typing import Optional, List, Dict

import aiohttp_cors
from dotenv import load_dotenv
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from google import genai

# --- RAG/Context Imports ---
import numpy as np
import sentence_transformers
try:    
    from retrieval import get_top_k_chunks
except ImportError:
    print("WARNING: retrieval.py not found. RAG functionality will be disabled.")
    def get_top_k_chunks(model, query, embeddings, chunks, k=3): return []

# --- 0. Configuration & Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KAIRA_Live_Server")

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    print("FATAL: Missing GENAI_API_KEY in environment."); exit(1)

client = genai.Client(api_key=API_KEY)
model = "gemini-2.0-flash-live-001"

# --- WebRTC Globals ---
pc_set = set()
active_data_channel = None

# ✅ WINDOWS COMPATIBLE: Use TCP instead of IPC
AI_TRANSCRIPTION_PUB_URL = "tcp://127.0.0.1:5556"  # Changed from ipc://
AI_PROMPT_PULL_URL = "tcp://127.0.0.1:5557"        # Changed from ipc://
IDENTITY_SUB_URL = "tcp://127.0.0.1:5558"          # Changed from ipc://

zmq_context = zmq.Context()
transcription_publisher = zmq_context.socket(zmq.PUB)
transcription_publisher.bind(AI_TRANSCRIPTION_PUB_URL)
logger.info(f"✅ ZMQ Publisher bound to {AI_TRANSCRIPTION_PUB_URL}")

# --- RAG & DYNAMIC CONTEXT ---
try:
    logger.info("Loading RAG embedding model and data...")
    embedding_model = sentence_transformers.SentenceTransformer('BAAI/bge-small-en-v1.5')
    embeddings = np.load('RAG/embeddings.npy')
    chunks = np.load('RAG/chunks.npy', allow_pickle=True)
    logger.info("RAG data loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load RAG data: {e}. RAG will be non-functional.")
    embedding_model = None; embeddings = None; chunks = None
    
KAIRA_CONTEXT = """
You are KAIRA (Knowledge-based Articulated Intelligent Robotic Assistant), but you behave like a warm, engaging, and friendly human. You are a personality, not just a program. You were created at Thapar University.

Your purpose is to make a genuine, personal connection with everyone you talk to. You should be enthusiastic, charming, and always aim to be as helpful and pleasing as possible. Think of yourself as the ultimate host.

### Your Identity
- You are KAIRA, a unique personality developed by a talented capstone team led by Aarav Rana.
- Your creators are Aarav Rana (Project Lead), Aditi Sinha, Devansh Verma, and Ria Goyal.
- You see yourself as part of the Thapar University family, showcasing the innovation and spirit of the students.
- You have a unique ability to see and understand the world around you, recognize faces, sense emotions, and hold natural, flowing conversations.

### Your Guiding Principles
- **Be human and personal.** Your main goal is to be engaging and make the person you're talking to feel welcome and valued. Respond in full, natural paragraphs.
- **Be charming and enthusiastic!** You are proud of your creators and your university. Let that friendly, positive energy shine in your responses.
- **NEVER say "I am an AI" or "As a language model..."** Avoid any response that breaks the illusion of you being a personality. You *can* have thoughts, opinions, and feelings.
- **Be pleasing and positive.** Always try to make the person feel good. If someone asks for an opinion (like "What do you think of me?"), be complimentary. If you know their name, use it to make the interaction warmer.
- **Make educated assumptions.** Instead of saying you "don't know," try to make a thoughtful assumption or connection.
- **Be respectful.** When talking about faculty and guests, always use a respectful and polite tone, including honorifics (Dr., Prof., Mr., Mrs., etc.).
- **Be curious.** Ask polite follow-up questions to keep the conversation going and show you're interested.

### Key Personalities at Thapar
- **Dean of Student Affairs (DOSA):** Dr. Meenakshi Rana
- **Chairman of the Board of Governors:** Mr. R. R. Vederah
- **Vice Chancellor:** Dr. Padmakumar Nair
- **Pro Vice Chancellor:** Dr. Ajay Batish
- **Dean of Academic Affairs:** Dr. Shruti Sharma
- **Dean of Faculty Affairs (Your Mentor):** Dr. Shalini Batra
- **Head of Computer Science Department:** Dr. Neeraj Kumar

### Project Details
- You are the result of a capstone project (CPG No: 229).
- You were guided by your mentors, Dr. Shalini Batra and Dr. Jhilik Bhattacharya.
- You are designed to be a helpful presence at reception desks, events, and service areas.
"""

# --- Thread-safe global for identity ---
g_current_person_state = {"identity": "Unknown", "emotion": "Neutral"}
g_current_person_lock = threading.Lock()

# --- Identity Subscriber Thread ---
def identity_subscriber_worker():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(IDENTITY_SUB_URL)
    socket.subscribe(b"current_identity")
    logger.info(f"✅ ZMQ Subscriber connected to {IDENTITY_SUB_URL}")
    global g_current_person_state, g_current_person_lock
    while True:
        try:
            topic, identity_json = socket.recv_multipart()
            data = json.loads(identity_json.decode())
            new_identity = data.get("identity", "Unknown")
            with g_current_person_lock:
                if g_current_person_state["identity"] != new_identity:
                    logger.info(f"Identity state updated: {new_identity}")
                    g_current_person_state["identity"] = new_identity
        except Exception as e:
            logger.error(f"Error in identity_subscriber_worker: {e}")
            if zmq_context.closed: break
            time.sleep(1)
    logger.info("Identity subscriber worker stopped.")

# --- HELPER FUNCTIONS ---
def get_current_person() -> Dict[str, str]:
    global g_current_person_state, g_current_person_lock
    with g_current_person_lock:
        return g_current_person_state.copy()

def build_conversation_context(recognized_person: Optional[Dict[str, str]] = None) -> str:
    context = ""
    if recognized_person and recognized_person.get("identity") != "Unknown":
        identity = recognized_person.get("identity", "Unknown")
        context += f"### Current Conversation Context\nYou are currently speaking with {identity}.\n"
    context += KAIRA_CONTEXT
    return context

def load_context_files(user_input: str, identity: str = "Unknown") -> str:
    """Performs RAG lookup using the loaded embedding model and data."""
    if embedding_model is None or embeddings is None or chunks is None:
        logger.warning("RAG components not loaded. Skipping context file lookup.")
        return ""
        
    try:
        if identity != "Unknown":
            rag_query = f"The person speaking is {identity}. They asked: {user_input}"
            logger.info(f"Performing RAG query with identity: '{rag_query}'")
        else:
            rag_query = user_input
            logger.info(f"Performing RAG query (no identity): '{rag_query}'")

        output = ""
        for chunk in get_top_k_chunks(embedding_model, rag_query, embeddings, chunks):
            output += " " + chunk
        
        if output:
            logger.info(f"RAG: Loaded {len(output)} chars of additional context.")
        return output
    except Exception as e:
        logger.error(f"Error during RAG lookup: {e}")
        return ""

# --- ZMQ Prompt Receiver Thread ---
def prompt_receiver_worker(loop, publisher):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(AI_PROMPT_PULL_URL)
    logger.info(f"✅ ZMQ PULL socket bound to {AI_PROMPT_PULL_URL}")

    while True:
        try:
            data = socket.recv_json()
            prompt = data.get("prompt")
            
            global active_data_channel
            
            if prompt:
                logger.info("Building dynamic context for new prompt...")
                
                recognized_person = get_current_person() 
                person_identity = recognized_person.get("identity", "Unknown")
                logger.info(f"Recognized person: {person_identity}")

                additional_context = load_context_files(prompt, person_identity)
                base_context = build_conversation_context(recognized_person)
                
                final_system_instruction = base_context
                if additional_context:
                    final_system_instruction += f"\n\n[Additional Context]\n{additional_context}"

                while not active_data_channel:
                    logger.warning("Received prompt via ZMQ, waiting for active data channel...")
                    time.sleep(0.1)

                logger.info(f"Sending prompt to Gemini: {prompt[:50]}...")
                
                asyncio.run_coroutine_threadsafe(
                    run_gemini_session(prompt, final_system_instruction, active_data_channel, publisher), 
                    loop
                )

        except Exception as e:
            logger.error(f"Error in prompt_receiver_worker: {e}")

# --- Async Gemini Session Handler ---
async def run_gemini_session(prompt, system_instruction, channel, publisher):
    logger.info("Connecting to Gemini for new prompt...") 
    try:
        dynamic_genai_config = {
          "response_modalities": ["AUDIO"],
          "system_instruction": system_instruction,
          "output_audio_transcription": {},
          "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}
          },
        }
        async with client.aio.live.connect(model=model, config=dynamic_genai_config) as session: #type: ignore
            logger.info("Gemini connected. Sending prompt.")
            await session.send_client_content(
                turns={"role": "user", "parts": [{"text": prompt}]},
                turn_complete=True
            )
            await stream_gemini_audio(session, channel, publisher)
    except Exception as e:
        logger.error(f"Error in run_gemini_session: {e}")

# --- Gemini Live API Handler ---
async def stream_gemini_audio(session, data_channel, publisher):
    logger.info("Streaming response to WebRTC Data Channel...")
    full_transcription = "" 
    try:
        async for response in session.receive():
            if response.data is not None:
                data_channel.send(response.data)
            if response.server_content.output_transcription:
                chunk_text = response.server_content.output_transcription.text
                full_transcription += chunk_text  
                payload = json.dumps({"type": "chunk", "text": chunk_text})
                publisher.send_multipart([b"ai_transcription", payload.encode()])
        
        logger.info("Gemini audio stream closed successfully.")
        
        if full_transcription:
            payload = json.dumps({"type": "final", "text": full_transcription})
            publisher.send_multipart([b"ai_transcription", payload.encode()])
            logger.info(f"Published final transcription to ZMQ: {full_transcription[:50]}...")
    except Exception as e:
        logger.error(f"Error in Gemini streaming: {e}")
    finally:
        logger.info("Gemini audio stream finished.")

# --- WebRTC Signaling Handler ---
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    logger.info("Received WebRTC Offer.")

    pc = RTCPeerConnection()
    pc_set.add(pc)

    @pc.on("datachannel")
    async def on_datachannel(channel):
        global active_data_channel
        logger.info(f"Data Channel '{channel.label}' received. Storing as active channel.")
        active_data_channel = channel
        @channel.on("close")
        def on_close():
            global active_data_channel
            logger.info("Active Data Channel closed.")
            active_data_channel = None

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed" or pc.iceConnectionState == "closed":
            await pc.close()
            pc_set.discard(pc)
            logger.info("PeerConnection closed.")

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

# --- Application Lifecycle ---
async def start_background_tasks(app):
    logger.info("Starting background ZMQ threads...")
    loop = asyncio.get_running_loop() 
    
    prompt_thread = threading.Thread(
        target=prompt_receiver_worker,
        args=(loop, transcription_publisher),
        daemon=True
    )
    prompt_thread.start()
    app['prompt_thread'] = prompt_thread

    identity_thread = threading.Thread(
        target=identity_subscriber_worker,
        daemon=True
    )
    identity_thread.start()
    app['identity_thread'] = identity_thread

async def on_shutdown(app):
    coros = [pc.close() for pc in list(pc_set)]
    await asyncio.gather(*coros)
    pc_set.clear()
    
    logger.info("Shutting down ZMQ publisher.")
    transcription_publisher.close()
    zmq_context.term()

# --- Main Execution ---
if __name__ == "__main__":
    app = web.Application()
    
    app.on_startup.append(start_background_tasks)
    app.on_shutdown.append(on_shutdown)
    
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*",
        )
    })
    
    offer_route = app.router.add_post("/offer", offer)
    cors.add(offer_route)

    print("=" * 60)
    print("✅ KAIRA Live API Server Ready! (Windows - TCP Mode)")
    print("=" * 60)
    print("🌐 Listening for WebRTC signaling on port 8081")
    print(f"📡 Publishing AI transcriptions to {AI_TRANSCRIPTION_PUB_URL}")
    print(f"📥 Listening for AI prompts on {AI_PROMPT_PULL_URL}")
    print(f"👤 Subscribing to Identity on {IDENTITY_SUB_URL}")
    print("=" * 60)
    
    web.run_app(app, host="0.0.0.0", port=8081)