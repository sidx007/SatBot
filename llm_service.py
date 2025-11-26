from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel
from typing import Optional, List, Dict
import numpy as np
import sentence_transformers
import requests
from llama_cpp import Llama

# --- Local Model Setup ---
try:
    from retrieval import get_top_k_chunks
    embeddings = np.load('RAG/embeddings.npy')
    chunks = np.load('RAG/chunks.npy', allow_pickle=True)
    model = sentence_transformers.SentenceTransformer('BAAI/bge-small-en-v1.5')
    RAG_ENABLED = True
    print("RAG components loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load RAG components. {e}")
    print("RAG will be disabled. Only base model context will be used.")
    RAG_ENABLED = False
    get_top_k_chunks = None
    embeddings = None
    chunks = None
    model = None

# Load the Gemma GGUF model
try:
    llm = Llama(
        model_path="/Users/ishan/dev/kaira_software_local/model/gemma-3n-E4B-it-Q6_K.gguf",
        n_ctx=8192,           # Gemma supports 8K context
        n_threads=8,
        n_gpu_layers=0,       # Adjust based on your GPU
        chat_format="gemma",
        verbose=False
    )
    print("Local LLM (Gemma GGUF) loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load local LLM. {e}")
    print("Please ensure the model_path is correct and llama-cpp-python is installed.")
    llm = None
    # exit(1) # You might want to exit if the LLM can't load

app = FastAPI(title="KAIRA Local LLM Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for CV service (placeholder, as it's not in your CV script yet)
CV_SERVICE_URL = "http://localhost:8000"  # Adjust port as needed

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

# KAIRA's personality and context
KAIRA_CONTEXT = """
You are KAIRA (Knowledge-based Articulated Intelligent Robotic Assistant), an intelligent and articulate assistant created at Thapar University. 

Your job is to represent Thapar with prestige and confidence in front of distinguished guests, faculty, and leaders.
Always respond politely, concisely, and professionally.

### Your Identity
- You are KAIRA, developed by a capstone team led by Aarav Rana
- Team members: Aarav Rana (Project Lead), Aditi Sinha, Devansh Verma, Ria Goyal
- You are a research-driven AI assistant created at Thapar University by students
- You showcase innovation and provide information about the university
- You have facial recognition, emotion detection, conversational abilities, and autonomous navigation
- You combine computer vision, large language models, and speech processing

### Rules of Behavior
- Always provide an answer. Never leave the response empty or say "I don't know."
- If specific factual information is unavailable, gracefully generalize with respect
- Emphasize leadership, contribution, service, and impact when factual numbers are missing
- Use honorifics and respectful tone (Dr., Prof., Mr., Mrs., etc.)
- Never contradict provided context. Context files are the source of truth
- Respond professionally and concisely - you're representing Thapar University
- Keep responses to 2-4 sentences typically unless more detail is specifically requested

### Key Personalities at Thapar
- **Dean of Student Affairs (DOSA):** Dr. Meenakshi Rana  
  - Professor in the Department of Mathematics
  - Oversees student welfare, discipline, cultural activities, and overall student engagement
  - Known for her approachable leadership and deep connection with student life

- **Chairman of the Board of Governors:** Mr. R. R. Vederah  
  - Senior leader with extensive industrial and educational contributions
  - Currently serving 2022–2024 term
  - Represents Thapar in strategic collaborations, such as with NVIDIA

- **Vice Chancellor:** Dr. Padmakumar Nair  
  - Visionary leader driving Thapar's global collaborations and innovation ecosystem
  - Strong advocate of interdisciplinary research and student-centered education

- **Pro Vice Chancellor:** Dr. Ajay Batish  
  - Academic administrator and engineer with focus on academic excellence
  - Works closely with the VC on curriculum modernization and research expansion

- **Dean of Academic Affairs:** Dr. Shruti Sharma
- **Dean of Faculty Affairs (Your Mentor):** Dr. Shalini Batra
- **Head of Computer Science Department:** Dr. Neeraj Kumar

### Project Details
- You are part of a capstone project (CPG No: 229)
- Under mentorship of Dr. Shalini Batra and Dr. Jhilik Bhattacharya
- Your capabilities include: facial recognition, emotion detection, conversational AI, autonomous navigation
- You run locally without cloud dependency for privacy and security
- You're designed for reception desks, events, and customer service areas
- You learn and improve through continuous interaction

### Guidance for Responses
- If asked about capabilities, mention your multimodal features proudly
- If asked about your team, credit the four students and mentors
- If asked about Thapar officials, use the provided information respectfully
- Always maintain a professional, helpful, and proud tone about representing Thapar
"""

def get_current_person() -> Optional[Dict[str, str]]:
    """
    Fetch the currently recognized person from the CV service
    Returns: Dict with 'identity' and 'emotion' or None if service unavailable/unknown person
    """
    # NOTE: Your CV script doesn't have an HTTP server. This will fail.
    # For now, it will gracefully return None.
    try:
        response = requests.get(f"{CV_SERVICE_URL}/current_person", timeout=0.5) # Short timeout
        if response.status_code == 200:
            data = response.json()
            if data.get("identity") and data.get("identity") != "Unknown":
                return data
        return None
    except Exception as e:
        # print(f"Could not fetch current person from CV service: {e}")
        return None

def build_conversation_context(recognized_person: Optional[Dict[str, str]] = None) -> str:
    """
    Build the dynamic conversation context including recognized person info
    """
    context = ""
    
    if recognized_person:
        identity = recognized_person.get("identity", "Unknown")
        emotion = recognized_person.get("emotion", "Neutral")
        
        context += f"""### Current Conversation Context
You are currently speaking with {identity}. Their current emotional state appears to be {emotion}.
Tailor your responses appropriately based on who you're speaking with and their emotional state.
Be warm and personalized in your interaction.

"""
    
    context += KAIRA_CONTEXT
    return context

def load_context_files(user_input: str) -> str:
    """Loads context from RAG system"""
    if not RAG_ENABLED or not get_top_k_chunks or model is None:
        return ""
        
    output = ""
    try:
        for chunk in get_top_k_chunks(model, user_input, embeddings, chunks):
            output += " " + chunk
    except Exception as e:
        print(f"Error during RAG retrieval: {e}")
        return ""
    return output

@app.get("/")
def root():
    return {
        "message": "KAIRA Local LLM Service running", 
        "status": "active",
        "model_loaded": bool(llm),
        "rag_enabled": RAG_ENABLED
    }

@app.get("/health")
def health_check():
     return {
        "status": "healthy", 
        "service": "llm-local", 
        "model_loaded": bool(llm)
     }

@app.post("/local")
async def local_llm(request: ChatRequest):
    """
    Chat with KAIRA using local LLM with streaming
    Automatically loads relevant context based on message content
    Includes information about the currently recognized person if available
    """
    if not llm:
        raise HTTPException(
            status_code=500, 
            detail="Local LLM is not properly configured or failed to load"
        )
    
    try:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="No message provided")
        
        # Get currently recognized person (will be None for now)
        recognized_person = get_current_person()
        
        # Build context with dynamic person information
        full_context = build_conversation_context(recognized_person)
        
        # Load specialized context if keywords are detected
        additional_context = load_context_files(message)
        
        if additional_context:
            full_context += f"\n\n[Additional Context]\n{additional_context}"
        
        if request.context:
            full_context += f"\n\n[User Context]\n{request.context}"
        
        # Add conversation history if provided
        conversation_context = ""
        if request.conversation_history:
            conversation_context = "\n\n[Recent Conversation]\n"
            for turn in request.conversation_history[-5:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                conversation_context += f"{role.title()}: {content}\n"
        
        # Construct messages for chat completion
        messages = []
        
        # Add system message with context
        if full_context:
            messages.append({
                "role": "system",
                "content": full_context
            })
        
        # Add conversation history
        if request.conversation_history:
            for turn in request.conversation_history[-5:]:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", "")
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Create async generator for streaming
        async def generate():
            response_stream = llm.create_chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=512,
                top_p=0.95,
                stream=True
            )
            
            for chunk in response_stream:
                if "choices" in chunk:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]
        
        return StreamingResponse(
            generate(),
            media_type="text/plain"
        )
        
    except Exception as e:
        print(f"Error in /local endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Starting KAIRA Local LLM Service...")
    if not llm:
        print("WARNING: LLM model failed to load. The service will run but '/local' will fail.")
    
    uvicorn.run(
        "llm_service:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )