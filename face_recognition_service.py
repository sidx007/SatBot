"""
Face Recognition Service - Windows Compatible
Receives camera frames and publishes identity information
"""

import zmq
import json
import numpy as np
import time
import cv2

# ✅ WINDOWS COMPATIBLE: TCP sockets
CAMERA_STREAM_URL = "tcp://127.0.0.1:5555"
IDENTITY_PUB_URL = "tcp://127.0.0.1:5558"

# Import your face recognition function
try:
    from identity_processing import process_identity_from_frame
    print("✅ Loaded identity_processing module")
except ImportError:
    print("⚠️  identity_processing.py not found, using mock recognition")
    def process_identity_from_frame(frame):
        """Mock function for testing"""
        return "Unknown"

def main():
    print("=" * 60)
    print("👤 KAIRA FACE RECOGNITION SERVICE (Windows)")
    print("=" * 60)
    print()
    
    # Setup ZMQ
    context = zmq.Context()
    
    # Subscribe to camera stream
    camera_sub = context.socket(zmq.SUB)
    camera_sub.connect(CAMERA_STREAM_URL)
    camera_sub.subscribe(b"camera_0")
    print(f"📹 Subscribed to camera stream: {CAMERA_STREAM_URL}")
    
    # Publish identity
    identity_pub = context.socket(zmq.PUB)
    identity_pub.bind(IDENTITY_PUB_URL)
    print(f"📡 Publishing identity to: {IDENTITY_PUB_URL}")
    
    print()
    print("🎬 Starting face recognition loop...")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    print()
    
    last_identity = "Unknown"
    frame_count = 0
    last_log_time = time.time()
    
    try:
        while True:
            # Receive frame from camera stream
            topic, meta_json, frame_data = camera_sub.recv_multipart()
            meta = json.loads(meta_json.decode())
            
            # Reconstruct frame
            frame = np.frombuffer(frame_data, dtype=meta['dtype'])
            frame = frame.reshape(meta['shape'])
            
            frame_count += 1
            
            # Process face recognition (every 10 frames to reduce load)
            if frame_count % 10 == 0:
                identity = process_identity_from_frame(frame)
                
                # Only publish if identity changed
                if identity != last_identity:
                    print(f"👤 Identity changed: {last_identity} → {identity}")
                    last_identity = identity
                    
                    # Publish identity update
                    identity_data = {
                        "identity": identity,
                        "emotion": "Neutral",  # Add emotion detection if available
                        "timestamp": time.time()
                    }
                    
                    identity_pub.send_multipart([
                        b"current_identity",
                        json.dumps(identity_data).encode()
                    ])
            
            # Log status every 5 seconds
            if time.time() - last_log_time > 5:
                print(f"✅ Processed {frame_count} frames | Current: {last_identity}")
                last_log_time = time.time()
                
    except KeyboardInterrupt:
        print("\n⚡ Stopping face recognition service...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera_sub.close()
        identity_pub.close()
        context.term()
        print("✅ Face recognition service stopped")

if __name__ == "__main__":
    main()
