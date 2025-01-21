#!/usr/bin/env python3
import os
import cv2
import sys
import json
import uuid
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
from av import VideoFrame

import time

from fractions import Fraction
import numpy as np

from pyvirtualdisplay import Display

import mss
from pydantic import BaseModel, Field  

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

from Xlib import X, XK, display
from Xlib.ext import xtest
from Xlib.protocol import event
import json

class MousePositionData(BaseModel):
    x: float = Field(..., description="X coordinate (0-1)")
    y: float = Field(..., description="Y coordinate (0-1)")
    type: str = Field(..., description="Event type (mousemove, mousedown, mouseup)")
    button: Optional[int] = Field(default=1, description="Mouse button number")


class InputData(BaseModel):
    type: str
    x: float = None
    y: float = None
    key: str = None
    button: int = None
    

GAME_WINDOW_WIDTH = 1920  
GAME_WINDOW_HEIGHT = 1080 
STREAM_WIDTH = 1280      
STREAM_HEIGHT = 720  

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Configuration
GAME_PATH = Path("cloud-game/sixcats/Six_Cats_Under.x86_64")
MAX_SESSIONS = 10


# HTML Template for the client
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Cloud Game Streaming</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
            color: #333;
        }
        .container {
            max-width: 1300px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        #status {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #e9ecef;
            border-radius: 4px;
            text-align: center;
        }
        #gameStream {
            width: 1280px;
            height: 720px;
            background: #000;
            margin: 0 auto;
            display: block;
            border-radius: 4px;
        }
        .controls {
            margin-top: 20px;
            text-align: center;
        }
        button {
            padding: 10px 20px;
            margin: 0 10px;
            border: none;
            border-radius: 4px;
            background-color: #007bff;
            color: white;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        .error {
            color: #dc3545;
            padding: 10px;
            margin: 10px 0;
            background-color: #f8d7da;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Cloud Game Streaming</h1>
        <div id="status">Connecting to game server...</div>
        <video id="gameStream" autoplay playsinline></video>
        <div class="controls">
            <button onclick="startGame()">Start Game</button>
            <button onclick="checkStatus()">Check Status</button>
        </div>
        <div id="error" class="error" style="display: none;"></div>
    </div>

    <script>
        const statusDiv = document.getElementById('status');
        const errorDiv = document.getElementById('error');
        const videoElement = document.getElementById('gameStream');
        let peerConnection = null;
        let websocket = null;
        let inputChannel = null;
        const video = document.querySelector('video');

        function setupInputHandlers() {
    if (!peerConnection) return;
    
    inputChannel = peerConnection.createDataChannel('input');
    inputChannel.onopen = () => console.log('Input channel opened');
    inputChannel.onclose = () => console.log('Input channel closed');
    
    // Keyboard input handling
    document.addEventListener('keydown', (e) => {
        if (!inputChannel || inputChannel.readyState !== 'open') return;
        inputChannel.send(JSON.stringify({
            type: 'keydown',
            key: e.key,
            code: e.code,
            timestamp: Date.now()
        }));
        // Prevent default for game controls
        if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) {
            e.preventDefault();
        }
    });

      document.addEventListener('keyup', (e) => {
        if (!inputChannel || inputChannel.readyState !== 'open') return;
        inputChannel.send(JSON.stringify({
            type: 'keyup',
            key: e.key,
            code: e.code,
            timestamp: Date.now()
        }));
    });

    // Mouse input handling
    const videoElement = document.getElementById('gameStream');
    
    videoElement.addEventListener('mousedown', (e) => {
        if (!inputChannel || inputChannel.readyState !== 'open') return;
        const rect = videoElement.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        inputChannel.send(JSON.stringify({
            type: 'mousedown',
            button: e.button,
            x: x,
            y: y,
            timestamp: Date.now()
        }));
    });

    videoElement.addEventListener('mouseup', (e) => {
        if (!inputChannel || inputChannel.readyState !== 'open') return;
        const rect = videoElement.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        inputChannel.send(JSON.stringify({
            type: 'mouseup',
            button: e.button,
            x: x,
            y: y,
            timestamp: Date.now()
        }));
    });

    videoElement.addEventListener('mousemove', (e) => {
        if (!inputChannel || inputChannel.readyState !== 'open') return;
        const rect = videoElement.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        inputChannel.send(JSON.stringify({
            type: 'mousemove',
            x: x,
            y: y,
            timestamp: Date.now()
        }));
    });

    // Prevent context menu on right-click
    videoElement.addEventListener('contextmenu', (e) => e.preventDefault());
}

        async function startGame() {
            try {
                statusDiv.textContent = 'Starting new game session...';
                const response = await fetch('/session', {
                    method: 'POST'
                });
                const data = await response.json();
                
                if (data.session_id) {
                    connectWebSocket(data.session_id);
                    statusDiv.textContent = 'Game session created. Connecting...';
                }
            } catch (error) {
                showError('Failed to start game session: ' + error.message);
            }
        }

        async function connectWebSocket(sessionId) {
            try {
                websocket = new WebSocket(`ws://${window.location.host}/ws/${sessionId}`);
                
                websocket.onopen = () => {
                    statusDiv.textContent = 'WebSocket connected. Setting up stream...';
                    setupWebRTC();
                };

                websocket.onmessage = async (event) => {
                    const message = JSON.parse(event.data);
                    if (message.type === 'answer') {
                        await peerConnection.setRemoteDescription(message);
                        statusDiv.textContent = 'Stream connected!';
                    }
                };

                websocket.onclose = () => {
                    statusDiv.textContent = 'Connection closed';
                    cleanup();
                };

                websocket.onerror = (error) => {
                    showError('WebSocket error: ' + error.message);
                    cleanup();
                };
            } catch (error) {
                showError('Failed to connect WebSocket: ' + error.message);
            }
        }

        async function setupWebRTC() {
            try {
                peerConnection = new RTCPeerConnection({
                    iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
                });

                 setupInputHandlers();

                peerConnection.ontrack = (event) => {
                    if (event.track.kind === 'video') {
                        videoElement.srcObject = new MediaStream([event.track]);
                    }
                };

                const offer = await peerConnection.createOffer({
                    offerToReceiveVideo: true,
                    offerToReceiveAudio: true
                });
                await peerConnection.setLocalDescription(offer);
                
                websocket.send(JSON.stringify({
                    type: offer.type,
                    sdp: offer.sdp
                }));
            } catch (error) {
                showError('Failed to setup WebRTC: ' + error.message);
            }
        }

        async function checkStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                statusDiv.textContent = `Server Status: ${data.status}`;
                if (data.error) {
                    showError(data.error);
                }
            } catch (error) {
                showError('Failed to check status: ' + error.message);
            }
        }

        function showError(message) {
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }

        function cleanup() {
            if (peerConnection) {
                peerConnection.close();
                peerConnection = null;
            }
            if (websocket) {
                websocket.close();
                websocket = null;
            }
        }

        // Check status on page load
        checkStatus();

        // Handle page unload
        window.onbeforeunload = () => {
            cleanup();
        };
        video.addEventListener('mousemove', (event) => {
    const rect = video.getBoundingClientRect();
    const x = Math.round(event.clientX - rect.left);
    const y = Math.round(event.clientY - rect.top);
    
    fetch('/mouse_position', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        x: 0.5,  // normalized coordinate (0-1)
        y: 0.5,  // normalized coordinate (0-1)
        type: 'mousemove',  // or 'mousedown' or 'mouseup'
        button: 1  // optional, defaults to 1
    })
});
});
    </script>
</body>
</html>
"""
# Video capture class


# Replace the GameVideoStreamTrack class with this implementation:

class GameVideoStreamTrack(VideoStreamTrack):
    def __init__(self, display_number):
        super().__init__()
        self.display_number = display_number
        
        logger.info(f"Initializing GameVideoStreamTrack for display :{display_number}")
        
        # No need to create a new display, use the existing one
        os.environ["DISPLAY"] = f":{display_number}"
        
        self._frame_counter = 0
        self._screen = mss.mss()
        
        # Configure to capture the virtual display
        self._monitor = {
            "top": 0,
            "left": 0,
            "width": 1280,
            "height": 720,
            "monitor": 1  # Usually monitor 1 is the virtual display
        }
        
        self._fps = 30
        self._time_base = Fraction(1, self._fps)
        self._last_frame_time = 0
        
        # Verify monitor configuration
        logger.info(f"Available monitors: {self._screen.monitors}")
        logger.info(f"Using monitor configuration: {self._monitor}")

    async def recv(self):
        """Capture and return next video frame"""
        try:
            # Frame rate control
            current_time = time.time()
            wait_time = (1.0 / self._fps) - (current_time - self._last_frame_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Capture frame
            screenshot = self._screen.grab(self._monitor)
            frame = np.array(screenshot)
            
            # Log frame info periodically
            if self._frame_counter % 30 == 0:
                logger.info(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
                # Add more detailed debug info
                if frame.size > 0:
                    logger.info(f"Frame min/max values: {frame.min()}/{frame.max()}")
                
            # Convert BGRA to BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Create video frame
            video_frame = VideoFrame.from_ndarray(frame_bgr, format='bgr24')
            video_frame.pts = self._frame_counter
            video_frame.time_base = self._time_base
            
            self._frame_counter += 1
            self._last_frame_time = current_time
            
            return video_frame
            
        except Exception as e:
            logger.error(f"Frame capture error: {str(e)}")
            logger.exception("Full traceback:")
            return await self._create_black_frame()
            
    async def _create_black_frame(self):
        """Create a black frame for error cases"""
        black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        video_frame = VideoFrame.from_ndarray(black_frame, format='bgr24')
        video_frame.pts = self._frame_counter
        video_frame.time_base = self._time_base
        self._frame_counter += 1
        return video_frame
    


class InputHandler:
    def __init__(self, display_number=0):
        self.display = None
        self.root = None
        self.display_number = display_number
        self.connect_to_display()

    def connect_to_display(self):
        """Connect to the X display and use the root window."""
        try:
            self.display = display.Display(f":{self.display_number}")
            self.root = self.display.screen().root
            logger.info(f"Connected to display :{self.display_number}")
        except Exception as e:
            logger.error(f"Display connection error: {e}")
            self.display = None
            self.root = None

    def send_mouse_event(self, event_type, x, y, button=1):
        """Send mouse events (movement, click) to the X server."""
        if not self.display or not self.root:
            if not self.connect_to_display():
                return False

        try:
            screen = self.display.screen()
            abs_x = max(0, min(int(x * GAME_WINDOW_WIDTH), GAME_WINDOW_WIDTH - 1))
            abs_y = max(0, min(int(y * GAME_WINDOW_HEIGHT), GAME_WINDOW_HEIGHT - 1))

            # Always move pointer first to ensure clicks happen at the right position
            xtest.fake_input(self.display, X.MotionNotify, 0, x=abs_x, y=abs_y)
            
            if event_type == "mousedown":
                xtest.fake_input(self.display, X.ButtonPress, button)
            elif event_type == "mouseup":
                xtest.fake_input(self.display, X.ButtonRelease, button)

            self.display.sync()
            self.display.flush()
            return True
        except Exception as e:
            logger.error(f"Error sending mouse event: {e}")
            return False
        
    def find_and_focus_window(self, window_name=None):
        """
        Find a window by name and set focus to it.
        If no name is provided, focuses the root window.
        """
        try:
            if not self.display:
                raise RuntimeError("Display not connected")

            root = self.display.screen().root
            window_ids = root.query_tree().children

            for window_id in window_ids:
                try:
                    window_name_current = window_id.get_wm_name()
                    if window_name and window_name_current == window_name:
                        window_id.set_input_focus(X.RevertToParent, X.CurrentTime)
                        logger.info(f"Focused window: {window_name}")
                        return True
                except Exception:
                    continue  # Skip windows without names

            # Default to focusing the root window
            self.root.set_input_focus(X.RevertToParent, X.CurrentTime)
            logger.info("Focused root window")
            return True
        except Exception as e:
            logger.error(f"Failed to focus window: {e}")
            return False
    def send_key_event(self, key_code, press):
        """Send keyboard event"""
        try:
            if not self.display or not self.root:
                if not self.connect_to_display():
                    return False

            # Ensure window has focus before sending key events
            self.find_and_focus_window()

            key_mapping = {
                'ArrowUp': XK.XK_Up,
                'ArrowDown': XK.XK_Down,
                'ArrowLeft': XK.XK_Left,
                'ArrowRight': XK.XK_Right,
                'Space': XK.XK_space,
                'Enter': XK.XK_Return,
                'Escape': XK.XK_Escape,
                # Handle both key codes and key values
                'KeyW': XK.XK_w,
                'w': XK.XK_w,
                'KeyA': XK.XK_a,
                'a': XK.XK_a,
                'KeyS': XK.XK_s,
                's': XK.XK_s,
                'KeyD': XK.XK_d,
                'd': XK.XK_d,
                'KeyE': XK.XK_e,
                'e': XK.XK_e,
                'KeyQ': XK.XK_q,
                'q': XK.XK_q
        
            }

            key_sym = key_mapping.get(key_code)
            if key_sym:
                keycode = self.display.keysym_to_keycode(key_sym)
                event_type = X.KeyPress if press else X.KeyRelease
                xtest.fake_input(self.display, event_type, keycode)
                self.display.sync()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Keyboard event error: {e}")
            return False
            
class GameSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.display = None
        self.process = None
        self.screen = None
        self.peer_connection = None
        self.video_track = None
        self.display_number = None
        self.input_handler = None


    async def setup_game(self):
        """Initialize game and virtual display"""
        try:
            # Start virtual display
            self.display = Display(visible=0, size=(1280, 720), backend='xvfb')
            self.display.start()
            self.display_number = self.display.display
            logger.info(f"Started virtual display :{self.display_number}")
            
            # Set display environment
            display_env = f":{self.display_number}"
            os.environ["DISPLAY"] = display_env
            
            # Ensure XAUTHORITY is set
            xauth_path = os.path.expanduser("~/.Xauthority")
            os.environ["XAUTHORITY"] = xauth_path
            
            # Wait for display to be ready
            await asyncio.sleep(2)
            
            # Launch game process with explicit display setting
            env = os.environ.copy()
            env["DISPLAY"] = display_env
            
            logger.info(f"Launching game with DISPLAY={display_env}")
            self.process = subprocess.Popen([
                str(GAME_PATH),
                "-screen-width", "1280",
                "-screen-height", "720",
                "-screen-fullscreen", "1"
            ], env=env)
            
            # Wait for game to start
            await asyncio.sleep(5)
            
            if self.process.poll() is not None:
                raise RuntimeError(f"Game process failed to start, exit code: {self.process.returncode}")
            
            logger.info(f"Game process started with PID: {self.process.pid}")
            
            # Initialize input handler
            self.input_handler = InputHandler(self.display_number)
            
            # Initial window focus
            success = self.input_handler.find_and_focus_window()
            if not success:
                logger.warning("Initial window focus failed, input may not work correctly")
            
            # Additional wait for window management
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Game setup error: {e}")
            await self.cleanup()
            raise

    async def handle_input(self, message):
        """Handle input messages from the client"""
        try:
            if not self.input_handler:
                self.input_handler = InputHandler(self.display_number)
            
            data = json.loads(message)
            event_type = data.get('type')
            
            if event_type in ('keydown', 'keyup'):
                self.input_handler.send_key_event(
                    data['code'],
                    event_type == 'keydown'
                )
            elif event_type in ('mousedown', 'mouseup', 'mousemove'):
                self.input_handler.send_mouse_event(
                    event_type,
                    data['x'],
                    data['y'],
                    data.get('button', 1)
                )
        except Exception as e:
            logger.error(f"Error handling input: {e}")


    async def setup_peer_connection(self, offer):
        """Setup WebRTC peer connection with video track"""
        try:
            logger.info("Creating new RTCPeerConnection")
            self.peer_connection = RTCPeerConnection()
            
            logger.info(f"Creating video track with display number {self.display_number}")
            self.video_track = GameVideoStreamTrack(self.display_number)
            self.peer_connection.addTrack(self.video_track)
            
            logger.info("Setting remote description")
            await self.peer_connection.setRemoteDescription(
                RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
            )
            
            logger.info("Creating answer")
            answer = await self.peer_connection.createAnswer()
            
            logger.info("Setting local description")
            await self.peer_connection.setLocalDescription(answer)
            
            return {
                "sdp": self.peer_connection.localDescription.sdp,
                "type": self.peer_connection.localDescription.type
            }
        except Exception as e:
            logger.error(f"Failed to setup peer connection: {e}")
            raise

    async def cleanup(self):
        """Clean up all resources"""
        logger.info("Starting cleanup...")
        
        if self.peer_connection:
            try:
                await self.peer_connection.close()
                logger.info("Closed peer connection")
            except Exception as e:
                logger.error(f"Error closing peer connection: {e}")
            
        if self.process:
            try:
                self.process.terminate()
                await asyncio.sleep(1)
                if self.process.poll() is None:
                    self.process.kill()
                logger.info(f"Game process terminated, exit code: {self.process.poll()}")
            except Exception as e:
                logger.error(f"Error terminating game process: {e}")
                
        if self.screen:
            try:
                self.screen.close()
                logger.info("Closed screen capture")
            except Exception as e:
                logger.error(f"Error closing screen capture: {e}")
                
        if self.display:
            try:
                self.display.stop()
                logger.info("Stopped virtual display")
            except Exception as e:
                logger.error(f"Error stopping virtual display: {e}")

def translate_coordinates(x, y):
    scale_x = GAME_WINDOW_WIDTH / STREAM_WIDTH
    scale_y = GAME_WINDOW_HEIGHT / STREAM_HEIGHT
    game_x = int(x * scale_x)
    game_y = int(y * scale_y)
    logger.debug(f"Translated coordinates: ({x}, {y}) -> ({game_x}, {game_y})")
    return game_x, game_y


class GameServer:
    def __init__(self):
        self.app = FastAPI()
        self.sessions: Dict[str, GameSession] = {}
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def home():
            return HTMLResponse(content=HTML_TEMPLATE)

        @self.app.post("/session")
        async def create_session():
            if len(self.sessions) >= MAX_SESSIONS:
                raise HTTPException(status_code=503, detail="Maximum sessions reached")
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = GameSession(session_id)
            return {"session_id": session_id}

        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            if session_id not in self.sessions:
                await websocket.close(code=4000)
                return

            session = self.sessions[session_id]
            await websocket.accept()

            try:
                # Setup game and get display number
                await session.setup_game()
                
                # Setup WebRTC
                offer = await websocket.receive_json()
                answer = await session.setup_peer_connection(offer)
                await websocket.send_json(answer)

                # Handle input messages
                while True:
                    try:
                        message = await websocket.receive_text()
                        await session.handle_input(message)
                    except json.JSONDecodeError:
                        logger.error("Invalid JSON received")
                        continue

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session {session_id}")
            except Exception as e:
                logger.error(f"Error in session {session_id}: {str(e)}")
            finally:
                await session.cleanup()
                if session_id in self.sessions:
                    del self.sessions[session_id]

        @self.app.get("/status")
        async def get_status():
            return {
                "status": "running",
                "active_sessions": len(self.sessions),
                "max_sessions": MAX_SESSIONS
            }

    def run(self, host="0.0.0.0", port=8000):
        uvicorn.run(self.app, host=host, port=port, log_level="info")

def main():
    logger.info("Starting Cloud Game Streaming Server...")
    
    if not GAME_PATH.exists():
        logger.warning(f"Game not found at {GAME_PATH}")
    
    server = GameServer()
    logger.info("Server starting on http://localhost:8000")
    uvicorn.run(server.app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()