# KAIRA - AI Voice Assistant with Animated Face

KAIRA (Keep AI Respectfully Accessible) is an intelligent voice assistant featuring a dynamic animated face that responds to speech and audio input. The system provides real-time speech-to-text, AI-powered responses, and text-to-speech with synchronized facial animations.

## 🏗️ System Architecture

KAIRA follows a microservices architecture with four main components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   STT Service   │    │   LLM Service   │    │   TTS Service   │
│   Port: 8001    │    │   Port: 8003    │    │   Port: 8004    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────▼────────┐             │
         └──────────────►│  Speaking Face  │◄────────────┘
                        │  (Main Client)  │
                        └─────────────────┘
```

## 📁 Project Structure

```
KAIRA---Testing-Module/
├── assets/                     # Visual assets for the animated face
│   ├── bg.svg                 # Background face template
│   ├── mouth [1-5].svg        # Different mouth positions for animation
│   ├── eye open big.svg       # Eye expressions
│   ├── eye close [up/down].svg# Eye blink animations
│   ├── glasses.svg            # Accessories
│   ├── blush.svg              # Emotional expressions
│   ├── meh.svg                # Neutral expression
│   ├── speak C.svg            # Speaking indicator
│   └── Flow [1-2].json        # Animation flow configurations
├── kaira_launcher.py          # Service orchestrator and launcher
├── speaking_face.py           # Main client with animated face
├── stt_service.py             # Speech-to-Text microservice
├── simple_tts_service.py      # Text-to-Speech microservice
├── llm_service.py             # AI Language Model microservice
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore patterns
└── README.md                  # This documentation
```

## 🔧 Core Components

### 1. **Speaking Face (`speaking_face.py`)** - Main Client Application

**Purpose**: The central orchestrator that provides the user interface and coordinates all services.

**Key Features**:
- **Real-time Audio Processing**: Captures microphone input using PyAudio
- **Visual Face Animation**: Renders SVG-based animated face using Pygame
- **Mouth Synchronization**: Syncs mouth movements with audio amplitude
- **Service Integration**: Communicates with all three microservices
- **User Controls**: Keyboard-based interaction (Space to record, Q to quit)

**Technical Details**:
```python
# Service URLs Configuration
self.stt_url = "http://localhost:8001"    # Speech-to-Text
self.tts_url = "http://localhost:8004"    # Text-to-Speech  
self.llm_url = "http://localhost:8003"    # Language Model

# Audio Configuration
CHUNK = 1024          # Audio buffer size
FORMAT = pyaudio.paFloat32  # 32-bit float audio
CHANNELS = 1          # Mono audio
RATE = 22050          # Sample rate (matches pygame mixer)
```

**Mouth Animation System**:
```python
# Amplitude-based mouth position mapping
self.mouth_thresholds = [
    (0.0, 0.1),   # closed mouth
    (0.1, 0.3),   # slightly open
    (0.3, 0.5),   # medium open
    (0.5, 0.7),   # wide open
    (0.7, 0.9),   # very wide
    (0.9, 1.0),   # maximum open
]
```

**Dependencies**:
- `pygame`: Graphics rendering and audio playback
- `numpy`: Audio signal processing
- `pyaudio`: Real-time audio I/O
- `requests`: HTTP communication with services

### 2. **Service Launcher (`kaira_launcher.py`)** - Orchestrator

**Purpose**: Automated service management and health monitoring.

**Key Features**:
- **Service Management**: Starts and stops all microservices
- **Health Monitoring**: Continuous service health checks
- **Dependency Validation**: Checks for required files and packages
- **Graceful Shutdown**: Handles cleanup on termination
- **Error Recovery**: Monitors and reports service failures

**Service Configuration**:
```python
services = [
    ("stt_service.py", "STT Service", 8001),
    ("simple_tts_service.py", "TTS Service", 8004),
    ("llm_service.py", "LLM Service", 8003)
]
```

**Health Check System**:
- Performs HTTP GET requests to `/health` endpoints
- 5-second timeout for service startup
- 10-second interval for continuous monitoring
- Automatic service restart on failure

### 3. **Speech-to-Text Service (`stt_service.py`)** - Port 8001

**Purpose**: Converts audio recordings to text using Google Speech Recognition.

**API Endpoints**:
- `GET /` - Service status
- `GET /health` - Health check
- `POST /transcribe` - Audio transcription

**Technical Implementation**:
```python
# Audio Processing Pipeline
1. Receive audio file via HTTP POST
2. Convert to AudioSegment using pydub
3. Export as WAV format
4. Process with SpeechRecognition library
5. Return transcribed text as JSON
```

**Supported Formats**:
- Input: WAV, MP3, FLAC, M4A
- Processing: Converts all to WAV internally
- Recognition: Google Speech Recognition API

**Error Handling**:
- Network timeout handling
- Audio format validation
- Recognition confidence scoring
- Fallback error messages

### 4. **Text-to-Speech Service (`simple_tts_service.py`)** - Port 8004

**Purpose**: Converts text to speech audio using Google Text-to-Speech.

**API Endpoints**:
- `GET /` - Service status  
- `GET /health` - Health check
- `POST /synthesize_realtime` - Text-to-speech conversion

**Audio Processing Pipeline**:
```python
# TTS Processing Flow
1. Receive text via HTTP POST
2. Generate MP3 using gTTS (Google TTS)
3. Convert MP3 to WAV using pydub
4. Optimize for pygame (22050Hz, stereo)
5. Return WAV audio data
```

**Technical Specifications**:
- **Input**: JSON with text field
- **Output**: WAV audio (22050Hz, 16-bit, stereo)
- **Language**: English (configurable)
- **Quality**: High-quality Google TTS voices

**Dependencies**:
- `gtts`: Google Text-to-Speech API
- `pydub`: Audio format conversion
- `fastapi`: REST API framework

### 5. **Language Model Service (`llm_service.py`)** - Port 8003

**Purpose**: Provides AI-powered conversational responses using Google Gemini.

**API Endpoints**:
- `GET /` - Service status
- `GET /health` - Health check  
- `POST /chat` - Conversational AI interaction

**AI Configuration**:
```python
# Gemini Model Settings
model = genai.GenerativeModel('gemini-1.5-flash')
temperature = 0.7        # Response creativity
max_tokens = 150        # Response length limit
```

**Features**:
- **Context Awareness**: Maintains conversation history
- **Personality**: Configured as helpful AI assistant
- **Response Filtering**: Ensures appropriate content
- **Error Handling**: Graceful fallback responses

## 🎨 Assets System

### Visual Components

The `assets/` directory contains SVG-based visual elements:

**Face Structure**:
- `bg.svg` - Base face outline and features
- `mouth [1-5].svg` - Progressive mouth opening states
- `eye open big.svg` - Alert/active eye state
- `eye close up/down.svg` - Blinking animations

**Expressions**:
- `glasses.svg` - Accessory overlay
- `blush.svg` - Emotional response indicator
- `meh.svg` - Neutral/thinking expression
- `speak C.svg` - Active speaking indicator

**Animation Flows**:
- `Flow 1.json` - Primary animation sequence
- `Flow 2.json` - Secondary animation patterns

### Animation System

The face animation is driven by real-time audio amplitude:

```python
# Real-time mouth sync algorithm
def update_mouth_position(self, amplitude):
    normalized_amplitude = min(amplitude * 10, 1.0)  # Normalize 0-1
    
    # Map amplitude to mouth position
    for i, (min_thresh, max_thresh) in enumerate(self.mouth_thresholds):
        if min_thresh <= normalized_amplitude < max_thresh:
            self.current_mouth_index = i
            break
```

## 🔄 Data Flow

### 1. **Voice Input Flow**
```
User speaks → Microphone → PyAudio → WAV buffer → STT Service → Text
```

### 2. **AI Processing Flow** 
```
Text → LLM Service → Gemini API → AI Response → Text
```

### 3. **Voice Output Flow**
```
Response Text → TTS Service → Google TTS → WAV Audio → Pygame → Speakers
```

### 4. **Visual Animation Flow**
```
Audio Input → Amplitude Analysis → Mouth Position → SVG Rendering → Display
```

## 🚀 Getting Started

### Prerequisites

**System Requirements**:
- Python 3.8+ 
- Windows/macOS/Linux
- Microphone and speakers
- Internet connection (for AI services)

**Required Packages**:
```bash
pip install -r requirements.txt
```

### Installation Steps

1. **Clone Repository**:
```bash
git clone https://github.com/Aaryan-549/KAIRA---Testing-Module.git
cd KAIRA---Testing-Module
```

2. **Set up Virtual Environment**:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux  
source venv/bin/activate
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure API Keys**:
Create environment variables or modify service files:
```bash
# For Google Gemini API
export GOOGLE_API_KEY="your_api_key_here"
```

5. **Start Services**:
```bash
# Option 1: Use launcher (recommended)
python kaira_launcher.py

# Option 2: Manual service startup
python stt_service.py &
python simple_tts_service.py &
python llm_service.py &
python speaking_face.py
```

### Usage

**Voice Interaction**:
- Press `SPACE` to start/stop recording
- Speak your question or command
- Watch KAIRA respond with voice and animation
- Press `Q` to quit

**Service Management**:
- Services auto-start with launcher
- Health monitoring ensures reliability
- Graceful shutdown with Ctrl+C

## 🔧 Configuration

### Audio Settings
```python
# In speaking_face.py
CHUNK = 1024          # Buffer size (lower = more responsive)
RATE = 22050          # Sample rate (higher = better quality)
CHANNELS = 1          # Mono/Stereo (1/2)
```

### Animation Tuning
```python
# Mouth sensitivity adjustment
self.mouth_thresholds = [
    (0.0, 0.1),   # Adjust thresholds for your microphone
    (0.1, 0.3),   # Lower values = more sensitive
    # ... more levels
]
```

### Service Ports
```python
# Default port configuration
STT_PORT = 8001       # Speech-to-Text
TTS_PORT = 8004       # Text-to-Speech
LLM_PORT = 8003       # Language Model
```

## 🧪 Development

### Adding New Features

**1. New Service Integration**:
```python
# Add to kaira_launcher.py services list
("new_service.py", "New Service", 8005)

# Add service URL to speaking_face.py
self.new_service_url = "http://localhost:8005"
```

**2. Animation Enhancements**:
- Add new SVG assets to `assets/` directory
- Update loading logic in `speaking_face.py`
- Modify animation triggers in `update_display()`

**3. Voice Customization**:
- Modify TTS service language/voice settings
- Add voice selection API endpoints
- Update UI controls for voice switching

### Debugging

**Service Health Check**:
```bash
# Test individual services
curl http://localhost:8001/health  # STT
curl http://localhost:8004/health  # TTS  
curl http://localhost:8003/health  # LLM
```

**Audio Debugging**:
```python
# Enable debug output in speaking_face.py
self.debug_audio = True  # Add this flag
# Prints amplitude values and mouth positions
```

**Network Debugging**:
```python
# Add request logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Comment complex audio/animation logic

### Pull Request Process
1. Fork the repository
2. Create feature branch (`feature/your-feature-name`)
3. Test thoroughly with all services
4. Update documentation if needed
5. Submit pull request with detailed description

### Testing
- Test voice recognition accuracy
- Verify animation synchronization
- Check service reliability under load
- Validate cross-platform compatibility

## 📊 Performance Optimization

### Audio Processing
- **Buffer Size**: Smaller chunks = lower latency, higher CPU usage
- **Sample Rate**: 22050Hz balances quality and performance
- **Audio Format**: Float32 provides good dynamic range

### Service Communication
- **Timeouts**: Configured for responsive user experience
- **Connection Pooling**: Reuse HTTP connections when possible
- **Error Recovery**: Graceful degradation on service failures

### Animation Rendering
- **Frame Rate**: Capped at 60 FPS for smooth animation
- **SVG Caching**: Pre-load assets to reduce rendering time
- **Amplitude Smoothing**: Prevents jittery mouth movements

## 🐛 Troubleshooting

### Common Issues

**1. Audio Input Problems**:
```bash
# Check microphone permissions
# Windows: Settings > Privacy > Microphone
# macOS: System Preferences > Security & Privacy > Microphone
```

**2. Service Connection Errors**:
```bash
# Check if services are running
netstat -an | findstr "8001 8003 8004"

# Restart services
python kaira_launcher.py
```

**3. Animation Not Working**:
- Verify assets directory exists and contains SVG files
- Check pygame installation and display capabilities
- Ensure proper audio input levels

**4. Poor Voice Recognition**:
- Check microphone quality and positioning
- Reduce background noise
- Speak clearly and at moderate pace
- Verify internet connection for Google services

### Log Analysis
- Service logs available in console output
- Check HTTP response codes for API issues
- Monitor CPU/memory usage during operation

## 📝 API Documentation

### STT Service API
```
POST /transcribe
Content-Type: multipart/form-data
Body: audio file

Response: {"text": "transcribed speech"}
```

### TTS Service API  
```
POST /synthesize_realtime
Content-Type: application/json
Body: {"text": "speech text"}

Response: WAV audio data
```

### LLM Service API
```
POST /chat
Content-Type: application/json  
Body: {"message": "user input", "conversation_id": "optional"}

Response: {"response": "AI reply", "conversation_id": "session_id"}
```

## 🔐 Security Considerations

- API keys should be stored as environment variables
- Service endpoints should be firewalled in production
- Audio data is processed locally when possible
- No persistent storage of voice recordings
- HTTPS recommended for production deployments

## 📈 Future Enhancements

### Planned Features
- **Multi-language Support**: Extend beyond English
- **Custom Voice Training**: Personal voice models
- **Advanced Emotions**: More facial expressions
- **Mobile App**: Companion mobile interface
- **Cloud Deployment**: Scalable cloud architecture

### Community Contributions Welcome
- Additional language models integration
- New animation styles and themes  
- Performance optimizations
- Accessibility improvements
- Extended platform support

---

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: Report bugs and request features
- **Discussions**: General questions and community support
- **Documentation**: This README and inline code comments

**Happy coding with KAIRA! 🤖✨**