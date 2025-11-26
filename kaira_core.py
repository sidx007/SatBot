import pyaudio
import numpy as np
import threading
import time
import logging
import zmq
import json
import queue
import os
from openwakeword.model import Model
from stt_processor import STTProcessor
from webrtc_client import WebRTCClient 

# --- Setup Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ZMQ URLs ---
AI_PROMPT_PUSH_URL = "tcp://127.0.0.1:5557"
AI_TRANSCRIPTION_SUB_URL = "tcp://127.0.0.1:5556"

class KAIRACore:
    def __init__(self):
        # --- State ---
        self.state = {
            'display_text': "",
            'kaira_response_text': "",
            'normalized_amplitude': 0.0,
            'is_final_sentence': False,
            'is_kaira_speaking': False,
            'last_sentence_time': 0,
            'listening_state': 'WAITING',
        }
        self.state_lock = threading.Lock()
        self.ai_response_timeout = 3.0  # 3 seconds
        self.ai_response_start_time = 0
        
        # --- Audio Settings ---
        self.chunk = 2048
        self.sample_rate = 16000
        self.channels = 1
        self.is_listening = False
        self.is_recording = False
        self.p_audio = pyaudio.PyAudio()
        self.input_stream = None

        # --- Wake Word Detection ---
        self.wake_word_threshold = 0.01  # Lowered from 3.5
        self.wake_word_model_path = os.path.join(os.path.dirname(__file__), "hey_kairaa.onnx")
        logger.info(f"Initializing wake word detector with model: {self.wake_word_model_path}")
        self.wake_word_detector = Model(wakeword_models=[self.wake_word_model_path], inference_framework='onnx')
        logger.info("Wake word detector initialized successfully")

        # --- STT Processor ---
        self.stt_processor = STTProcessor(
            on_realtime_text=self._on_stt_realtime,
            on_full_sentence_text=self._on_stt_full_sentence
        )
        
        # --- ZMQ Sockets ---
        logger.info("Initializing ZMQ sockets...")
        self.zmq_context = zmq.Context()
        self.prompt_pusher = self.zmq_context.socket(zmq.PUSH)
        self.prompt_pusher.connect(AI_PROMPT_PUSH_URL)
        self.transcription_sub = self.zmq_context.socket(zmq.SUB)
        self.transcription_sub.connect(AI_TRANSCRIPTION_SUB_URL)
        self.transcription_sub.subscribe(b"ai_transcription")
        
        self.transcription_thread = threading.Thread(
            target=self._transcription_subscriber_worker, 
            daemon=True
        )
        
        # --- Audio Playback & WebRTC ---
        self.audio_playback_queue = queue.Queue()
        self.playback_stream = None
        self.audio_playback_thread = threading.Thread(
            target=self._audio_playback_worker,
            daemon=True
        )
        self.webrtc_client = WebRTCClient(self.audio_playback_queue)
        
        print("KAIRA Core initialized.")

    def _audio_playback_worker(self):
        """Runs in a thread, initializes speaker output, and plays audio."""
        logger.info("Audio playback worker started.")
        self.playback_stream = self.p_audio.open(
            format=pyaudio.paInt16, 
            channels=1, 
            rate=24000, 
            output=True
        )
        while self.is_listening:
            try:
                chunk = self.audio_playback_queue.get(timeout=1.0)
                if chunk is None:
                    break
                if self.playback_stream:
                    self.playback_stream.write(chunk)
            except queue.Empty:
                continue
            except Exception as e:
                if self.is_listening:
                    logger.error(f"Audio playback error: {e}")
        
        logger.info("Audio playback worker stopping...")
        if self.playback_stream:
            self.playback_stream.stop_stream()
            self.playback_stream.close()
        logger.info("Audio playback worker stopped.")

    def _transcription_subscriber_worker(self):
        """Listens for AI text and updates the state."""
        logger.info(f"Listening for AI transcriptions on {AI_TRANSCRIPTION_SUB_URL}")
        while self.is_listening:
            try:
                topic, payload = self.transcription_sub.recv_multipart()
                data = json.loads(payload.decode())
                
                with self.state_lock:
                    if data['type'] == 'chunk':
                        text_chunk = data['text']
                        if not self.state['is_kaira_speaking']:
                            self.state['is_kaira_speaking'] = True
                            self.state['display_text'] = ""
                            self.state['is_final_sentence'] = False
                        self.state['kaira_response_text'] += text_chunk
                    elif data['type'] == 'final':
                        self.state['is_kaira_speaking'] = False
                        self.state['last_sentence_time'] = time.time()
            except zmq.ZMQError as e:
                if not self.is_listening:
                    break 
                logger.error(f"ZMQ Error in transcription worker: {e}")
            except Exception as e:
                logger.error(f"Error in transcription worker: {e}")
        
        logger.info("Transcription subscriber worker stopped.")

    def start_audio_input_stream(self):
        """Start the audio input stream for listening"""
        if not self.p_audio:
            logger.error("PyAudio not initialized.")
            return
        
        try:
            self.input_stream = self.p_audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self.audio_input_callback
            )
            self.input_stream.start_stream()
            logger.info(f"Audio input stream started (Rate: {self.sample_rate}Hz)")
        except Exception as e:
            logger.error(f"Error starting audio input stream: {e}")

    def audio_input_callback(self, in_data, frame_count, time_info, status):
        """Audio input callback to process wake word detection and feed STT"""
        try:
            # Convert audio data to numpy array for wake word detection
            audio_array = np.frombuffer(in_data, dtype=np.int16)

            # If currently recording, feed audio to STT processor
            if self.is_recording and self.stt_processor:
                self.stt_processor.feed_audio(in_data)
            # Otherwise, check for wake word
            elif not self.is_recording:
                # Predict wake word from audio chunk
                prediction = self.wake_word_detector.predict(audio_array)

                # Check if wake word was detected above threshold
                for model_name, score in prediction.items():
                    # Debug logging
                    if score > 0.1:
                        logger.debug(f"Wake word score: {score:.3f}")
                    
                    if score >= self.wake_word_threshold:
                        logger.info(f"Wake word detected! Score: {score:.2f}")
                        # Trigger recording
                        self.start_recording()
                        # Feed this audio chunk to STT immediately
                        if self.stt_processor:
                            self.stt_processor.feed_audio(in_data)
                        break

        except Exception as e:
            logger.error(f"Audio input callback error: {e}")
        
        return (in_data, pyaudio.paContinue)

    # --- STT Callbacks ---
    def _on_stt_realtime(self, text):
        """Realtime STT callback — handles normal text and voice stop command."""
        if not text:
            return

        normalized = text.lower().strip()

        # 🔴 STOP COMMAND via STT (works even in middle of speech)
        if "stop kaira" in normalized:
            logger.info("🛑 'Stop Kaira' detected in speech — halting listening.")
            self.stop_recording()
            with self.state_lock:
                self.state['display_text'] = ""
                self.state['is_final_sentence'] = True
                self.state['listening_state'] = "WAITING"
            return

        # 🧠 Normal real-time text handling during active listening
        with self.state_lock:
            if self.state['listening_state'] == 'LISTENING':
                self.state['display_text'] = text
                self.state['is_final_sentence'] = False


    def _on_stt_full_sentence(self, text):
        """
        Callback from STTProcessor for a full sentence.
        This sends the prompt AND automatically stops the recording.
        """
        with self.state_lock:
            self.state['display_text'] = text
            self.state['last_sentence_time'] = time.time()
            self.state['is_final_sentence'] = True
            self.state['is_kaira_speaking'] = True 
            self.state['kaira_response_text'] = ""
            self.ai_response_start_time = time.time()
        
        # Send prompt to AI
        try:
            logger.info(f"Sending prompt to AI: '{text}'")
            payload = {"prompt": text, "timestamp": time.time()}
            self.prompt_pusher.send_json(payload)
        except Exception as e:
            logger.error(f"Failed to send prompt via ZMQ: {e}")
            with self.state_lock:
                self.state['is_kaira_speaking'] = False
                self.state['kaira_response_text'] = "Error: Could not connect to AI."
        
        # Automatically stop recording (acts like VAD)
        self.stop_recording()

    # --- Recording Control Methods ---
    def start_recording(self):
        """Starts recording with guard clause to prevent interruption"""
        with self.state_lock:
            # Don't start if AI is speaking OR if already recording
            if self.state['is_kaira_speaking'] or self.is_recording:
                if self.state['is_kaira_speaking']:
                    logger.warning("Input blocked: KAIRA is still speaking.")
                return 
            
            logger.info("--- Wake Word DETECTED: Recording START ---")
            self.is_recording = True
            self.state['listening_state'] = 'LISTENING'
            self.state['display_text'] = "..." 
            self.state['is_final_sentence'] = False
            self.state['kaira_response_text'] = ""

    def stop_recording(self):
        """Stops recording"""
        if not self.is_recording:
            return

        logger.info("--- Recording STOP (auto-stopped) ---")
        self.is_recording = False
        with self.state_lock:
            self.state['listening_state'] = 'WAITING'

    # --- Public Methods ---
    def get_state(self):
        """Get a copy of the current state"""
        with self.state_lock:
            return self.state.copy()

    def start(self):
        """Start all core services"""
        print("Starting KAIRA Core services...")
        self.is_listening = True 
        self.stt_processor.start()
        self.transcription_thread.start()
        self.audio_playback_thread.start()
        self.webrtc_client.start()
        self.start_audio_input_stream()
        print("🚀 KAIRA Core is running (listening for wake word 'hey kaira').")

    def stop(self):
        """Stop all core services"""
        print("\n🧹 Stopping KAIRA Core services...")
        self.is_listening = False 
        
        if self.stt_processor:
            self.stt_processor.stop()
            
        self.webrtc_client.stop()
        self.audio_playback_queue.put(None) 
        self.audio_playback_thread.join(timeout=1.0)
            
        if self.transcription_thread:
            self.transcription_thread.join(timeout=1.0)
        
        self.prompt_pusher.close()
        self.transcription_sub.close()
        self.zmq_context.term()
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        
        if self.p_audio:
            self.p_audio.terminate()
        
        print("✅ KAIRA Core stopped.")