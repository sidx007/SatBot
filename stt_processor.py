# stt_processor.py

import threading
import logging
import sys
from RealtimeSTT import AudioToTextRecorder

# --- Basic Setup (Unchanged) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger('faster_whisper').setLevel(logging.WARNING)
logging.getLogger('RealtimeSTT').setLevel(logging.WARNING)

class STTProcessor:
    """
    Manages the RealtimeSTT recorder.
    Receives audio chunks manually.
    """
    
    # --- RECORDER CONFIG MODIFIED ---
    recorder_config = {
        'spinner': False,
        'use_microphone': False, # <-- ROLLED BACK
        # "input_device_index": 5,
        'model': 'small',
        'language': 'en',
        'silero_sensitivity': 0.5,
        'webrtc_sensitivity': 1,
        'post_speech_silence_duration': 0.5,
        'min_length_of_recording': 0,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0,
        'realtime_model_type': 'tiny.en',
        # --- Wake Word Config REMOVED ---
    }

    def __init__(self, on_realtime_text, on_full_sentence_text): # <-- Removed wakeword callback
        self.on_realtime_text = on_realtime_text
        self.on_full_sentence_text = on_full_sentence_text
        
        # --- Callbacks MODIFIED ---
        self.recorder_config['on_realtime_transcription_stabilized'] = self._realtime_callback
        # --- Removed wakeword callback ---
        
        logging.info("Initializing RealtimeSTT (manual audio feed)...")
        self.recorder = AudioToTextRecorder(**self.recorder_config)
        logging.info("RealtimeSTT initialized.")
        
        self.is_running = False
        self.processing_thread = None

    # --- wakeword_callback REMOVED ---

    def _realtime_callback(self, text):
        """Internal callback for RealtimeSTT (Unchanged)"""
        if self.on_realtime_text:
            self.on_realtime_text(text)

    def _processing_loop(self):
        """
        The main loop for the recorder thread.
        """
        logging.info("STT processing thread started.")
        while self.is_running:
            try:
                # This call will block until audio is fed
                # and then silence is detected (when user releases spacebar)
                full_sentence = self.recorder.text()
                
                if full_sentence and self.is_running:
                    logging.info(f"Detected full sentence: {full_sentence}")
                    if self.on_full_sentence_text:
                        self.on_full_sentence_text(full_sentence)
                
            except Exception as e:
                if self.is_running:
                    logging.error(f"Error in STT processing loop: {e}")
                else:
                    logging.info("STT loop interrupted by stop().")
                
        logging.info("STT processing thread stopped.")

    def start(self):
        """Starts the STT processing thread (Unchanged)"""
        if self.is_running:
            return
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logging.info("RealtimeSTT recorder is waiting for audio...")

    def stop(self):
        """Stops the STT processing thread (Unchanged)"""
        if not self.is_running:
            return
        logging.info("Stopping STT processor...")
        self.is_running = False
        if self.recorder:
            self.recorder.stop() 
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        logging.info("STT processor stopped.")

    # --- feed_audio method ADDED BACK ---
    def feed_audio(self, audio_chunk):
        """
        Feeds a raw audio chunk (bytes) to the recorder.
        """
        if self.is_running and audio_chunk:
            try:
                self.recorder.feed_audio(audio_chunk)
            except Exception as e:
                logging.error(f"Error feeding audio to STT: {e}")