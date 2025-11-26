import asyncio
import json
import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import threading
import queue

class WebcamTrack(VideoStreamTrack):
    """Video stream track that captures from a webcam."""
    
    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  # ✅ Windows: Use DirectShow
        
        if not self.cap.isOpened():
            raise Exception(f"Could not open video capture device {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.q = queue.Queue(maxsize=2)  # Small buffer for latest frames
        self.running = True

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        print(f"✅ Camera {self.camera_index} initialized successfully")

    def _reader(self):
        """Continuously reads frames from the camera in a separate thread."""
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                consecutive_failures += 1
                print(f"⚠️  Camera {self.camera_index} read failed ({consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    print(f"❌ Camera {self.camera_index} failed too many times. Stopping.")
                    break
                    
                # Wait a bit before retrying
                threading.Event().wait(0.1)
                continue
            
            # Reset failure counter on success
            consecutive_failures = 0
            
            # Clear old frame if queue is full
            while self.q.qsize() >= self.q.maxsize:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            
            # Put the new frame
            try:
                self.q.put(frame, block=False)
            except queue.Full:
                pass  # Skip frame if queue is full
        
        # Signal that the thread is stopping
        self.q.put(None)
        print(f"🛑 Camera {self.camera_index} reader thread stopped")

    async def recv(self):
        """Receive the next frame."""
        pts, time_base = await self.next_timestamp()
        
        # Get the latest frame from the queue
        try:
            frame = await asyncio.wait_for(
                asyncio.to_thread(self.q.get), 
                timeout=5.0
            )
        except asyncio.TimeoutError:
            print(f"⚠️  Timeout waiting for frame from camera {self.camera_index}")
            # Return black frame on timeout
            frame = None

        if frame is None:
            # Thread has stopped or timeout occurred, return black frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Convert to VideoFrame
        frame_av = VideoFrame.from_ndarray(frame, format="bgr24")
        frame_av.pts = pts
        frame_av.time_base = time_base
        return frame_av

    def stop(self):
        """Stop the camera capture."""
        if self.running:
            print(f"🛑 Stopping camera {self.camera_index}...")
            self.running = False
            if self.cap.isOpened():
                self.cap.release()
            if self.thread.is_alive():
                self.thread.join(timeout=2.0)
            print(f"✅ Camera {self.camera_index} stopped")

    def __del__(self):
        self.stop()


# Global set to track peer connections
pcs = set()

async def offer(request):
    """Handle WebRTC offer from client."""
    try:
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        print("📨 Received WebRTC offer")

        pc = RTCPeerConnection()
        pcs.add(pc)

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            print(f"🔗 ICE connection state: {pc.iceConnectionState}")
            if pc.iceConnectionState == "failed" or pc.iceConnectionState == "closed":
                await pc.close()
                pcs.discard(pc)
                print("🔌 Peer connection closed")

        # Create and add SINGLE webcam track
        try:
            webcam_track = WebcamTrack(camera_index=0)
            pc.addTrack(webcam_track)
            print("✅ Added camera track to peer connection")
        except Exception as e:
            print(f"❌ Failed to initialize camera: {e}")
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"error": f"Camera initialization failed: {str(e)}"})
            )

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        print("✅ Sending WebRTC answer")
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )
        
    except Exception as e:
        print(f"❌ Error handling offer: {e}")
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"error": str(e)})
        )

async def on_shutdown(app):
    """Clean up resources on shutdown."""
    print("🛑 Shutting down server...")
    coros = [pc.close() for pc in list(pcs)]
    await asyncio.gather(*coros, return_exceptions=True)
    pcs.clear()
    print("✅ All peer connections closed")

def handle_async_exception(loop, context):
    """Handle async exceptions gracefully."""
    exception = context.get("exception")
    
    if isinstance(exception, asyncio.exceptions.InvalidStateError):
        # This is a known harmless race condition in aioice
        pass
    else:
        message = context.get("message", "Unknown error")
        print(f"⚠️  Async exception: {message}")
        if exception:
            print(f"   Exception type: {type(exception).__name__}: {exception}")

if __name__ == "__main__":
    # Create application
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    
    # Set up custom exception handler
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(handle_async_exception)
    
    print("=" * 60)
    print("🎥 CAMERA SERVER (Windows)")
    print("=" * 60)
    print("📡 Serving single camera (index 0)")
    print("🌐 Listening on http://0.0.0.0:8080")
    print("🔌 WebRTC endpoint: POST /offer")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print()
    
    try:
        web.run_app(app, host="0.0.0.0", port=8080, loop=loop)
    except KeyboardInterrupt:
        print("\n⚡ Server interrupted by user")
    finally:
        print("✅ Server shutdown complete")