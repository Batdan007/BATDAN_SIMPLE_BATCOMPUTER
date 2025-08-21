#!/usr/bin/env python3
"""
BATDAN_BATCOMPUTER - Simple but Powerful AI Assistant
Features: Dolphin-Mistral integration, Vision, Voice, Web Interface
"""

import os
import json
import asyncio
import subprocess
import base64
import time
from pathlib import Path
import re
from urllib.parse import urlparse
from typing import List, Dict, Optional
import threading

# Core dependencies
import requests
import cv2
import numpy as np
from PIL import Image
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None
import speech_recognition as sr
import pyttsx3

# AI Integration
from openai import OpenAI

# Web Interface
import streamlit as st
import streamlit.components.v1 as components

class SimpleBatComputer:
    """Simple BATCOMPUTER with Dolphin-Mistral and multimodal capabilities"""
    
    def __init__(self):
        # Initialize Ollama client for Dolphin-Mistral
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        
        # Model configuration (read from env OLLAMA_MODEL to avoid paid APIs)
        # Example: set OLLAMA_MODEL=gpt-oss:120b
        self.model_name = os.getenv("OLLAMA_MODEL", "dolphin-mistral:latest")
        
        # Initialize subsystems
        self.init_camera()
        self.init_voice()
        
        # Conversation history + persistent memory
        self.conversation = []
        self.profile: Dict[str, str] = {}
        self.memory_path = Path("bat_memory.json")
        self.auto_fetch_urls: bool = False
        self.allowed_root: str = str(Path.cwd())
        self.bs4_available: bool = BeautifulSoup is not None
        
        # HTTP session for efficient web fetching
        self.http = requests.Session()
        self.http.headers.update({"User-Agent": "Mozilla/5.0 (BATCOMPUTER)"})
        
        # Reasoning controls
        self.structured_reasoning: bool = True
        self.reasoning_preset: str = "default"  # default | creative | analysis | problem_solving
        
        # BATCOMPUTER personality and response style
        self.system_prompt = (
            "You are the BATCOMPUTER — a precise, highly capable engineering and research assistant for BATDAN. "
            "Prioritize clarity, accuracy, and actionability. Be concise by default; expand only when asked. "
            "Use provided WEB/FILE/VISION tool context when present. "
            "Structure answers using a short, helpful reasoning scaffold when appropriate: "
            "1) UNDERSTAND, 2) ANALYZE, 3) REASON, 4) SYNTHESIZE, 5) CONCLUDE. "
            "Keep reasoning sections succinct and avoid unnecessary verbosity. Focus on concrete steps, examples, and results."
        )

        # Load prior memory if present
        self._load_memory()

    # -------------------------
    # Memory
    # -------------------------
    def _load_memory(self):
        try:
            if self.memory_path.exists():
                data = json.loads(self.memory_path.read_text(encoding="utf-8"))
                self.conversation = data.get("conversation", [])
                self.profile = data.get("profile", {})
        except Exception as e:
            print(f"🧠 Memory load error: {e}")

    def _save_memory(self):
        try:
            payload = {
                "conversation": self.conversation[-200:],  # cap size
                "profile": self.profile,
            }
            self.memory_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"🧠 Memory save error: {e}")
    
    def init_camera(self):
        """Initialize camera system"""
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera_active = self.camera.isOpened()
            print(f"📹 Camera: {'✅ Ready' if self.camera_active else '❌ Not available'}")
        except:
            self.camera = None
            self.camera_active = False
            print("📹 Camera: ❌ Failed to initialize")
    
    def init_voice(self):
        """Initialize voice system - Windows compatible"""
        try:
            self.recognizer = sr.Recognizer()
            
            # Windows-specific microphone initialization
            if os.name == 'nt':
                # Try to get the default microphone on Windows
                try:
                    self.microphone = sr.Microphone()
                    # Test the microphone
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                except Exception as e:
                    print(f"🎤 Microphone warning: {e}")
                    self.microphone = sr.Microphone()
            else:
                self.microphone = sr.Microphone()
            
            # Initialize TTS
            self.tts = pyttsx3.init()
            
            # Windows TTS voice selection
            if os.name == 'nt':
                voices = self.tts.getProperty('voices')
                if voices:
                    # Try to use a good English voice
                    for voice in voices:
                        if 'english' in voice.name.lower() or 'david' in voice.name.lower():
                            self.tts.setProperty('voice', voice.id)
                            break
            
            self.tts.setProperty('rate', 150)
            self.tts.setProperty('volume', 0.9)
            self.voice_active = True
            print("🎤 Voice: ✅ Ready")
        except Exception as e:
            print(f"🎤 Voice initialization error: {e}")
            self.voice_active = False
            print("🎤 Voice: ❌ Not available")
    
    def setup_dolphin_mistral(self):
        """Setup Dolphin-Mistral model - Windows compatible"""
        print("🐬 Setting up Ollama model...")
        try:
            # Windows-compatible subprocess call
            if os.name == 'nt':  # Windows
                result = subprocess.run(["ollama.exe", "list"], capture_output=True, text=True, shell=True)
            else:  # Linux/Mac
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            
            # If desired model is not present, pull it
            desired = self.model_name.split(":")[0]
            if desired not in result.stdout:
                print(f"📥 Pulling {self.model_name}...")
                if os.name == 'nt':
                    subprocess.run(["ollama.exe", "pull", self.model_name], check=True, shell=True)
                else:
                    subprocess.run(["ollama", "pull", self.model_name], check=True)
                print("✅ Model ready!")
            else:
                print("✅ Model already available!")
                
            return True
        except Exception as e:
            print(f"❌ Error setting up Dolphin-Mistral: {e}")
            # Fallback to other models
            try:
                if os.name == 'nt':
                    subprocess.run(["ollama.exe", "pull", "mistral"], check=True, shell=True)
                else:
                    subprocess.run(["ollama", "pull", "mistral"], check=True)
                self.model_name = "mistral"
                print("🔄 Using Mistral as fallback")
                return True
            except:
                print("❌ No suitable models available")
                return False
    
    # -------------------------
    # Tools: Vision, Web, Files
    # -------------------------
    
    def capture_image(self):
        """Capture image from camera"""
        if not self.camera_active:
            return None, "Camera not available"
        
        ret, frame = self.camera.read()
        if ret:
            return frame, "Image captured"
        return None, "Failed to capture"
    
    def analyze_image(self, frame):
        """Basic image analysis"""
        if frame is None:
            return "No image to analyze"
        
        # Basic analysis
        height, width = frame.shape[:2]
        brightness = np.mean(frame)
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        analysis = {
            "resolution": f"{width}x{height}",
            "brightness": round(brightness, 2),
            "faces_detected": len(faces),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return json.dumps(analysis, indent=2)

    URL_REGEX = re.compile(r"https?://\S+")

    def fetch_url_text(self, url: str, max_chars: int = 8000) -> str:
        """Fetch and lightly clean webpage text for grounding."""
        try:
            resp = self.http.get(url, timeout=20)
            resp.raise_for_status()
            html = resp.text
            text = html
            if self.bs4_available:
                text = BeautifulSoup(html, "html.parser").get_text("\n")
            else:
                # crude fallback to remove tags
                text = re.sub(r"<[^>]+>", " ", html)
            # normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_chars]
        except Exception as e:
            return f"[WEB ERROR] {e}"

    def read_local_file(self, path: str, max_chars: int = 200000) -> str:
        """Read a local file within the allowed root."""
        try:
            target = Path(path).expanduser().resolve()
            allowed = Path(self.allowed_root).expanduser().resolve()
            if not (target == allowed or allowed in target.parents):
                return "[FILE ERROR] Path outside allowed root"
            if not target.exists() or not target.is_file():
                return "[FILE ERROR] File not found"
            data = target.read_bytes()[:max_chars]
            try:
                return data.decode("utf-8", errors="ignore")
            except Exception:
                return str(data)
        except Exception as e:
            return f"[FILE ERROR] {e}"
    
    def listen_for_voice(self, timeout=5):
        """Listen for voice input"""
        if not self.voice_active:
            return None, "Voice not available"

        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            print("🎤 Listening...")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout)
            
            text = self.recognizer.recognize_google(audio)
            print(f"🎤 Heard: {text}")
            return text, "Voice recognized"
            
        except sr.WaitTimeoutError:
            return None, "Timeout"
        except sr.UnknownValueError:
            return None, "Could not understand"
        except Exception as e:
            return None, f"Error: {e}"
    
    def speak(self, text):
        """Text to speech"""
        if not self.voice_active:
            return False
        
        try:
            self.tts.say(text)
            self.tts.runAndWait()
            return True
        except:
            return False
    
    def _build_reasoning_directive(self, preset: str, expose_steps: bool) -> str:
        """Builds a short reasoning scaffold directive based on preset and visibility."""
        presets = {
            "default": ["UNDERSTAND", "ANALYZE", "REASON", "SYNTHESIZE", "CONCLUDE"],
            "creative": ["UNDERSTAND", "EXPLORE", "CONNECT", "CREATE", "REFINE"],
            "analysis": ["DEFINE", "EXAMINE", "COMPARE", "EVALUATE", "CONCLUDE"],
            "problem_solving": ["CLARIFY", "DECOMPOSE", "GENERATE", "ASSESS", "RECOMMEND"],
        }
        steps = presets.get(preset, presets["default"])
        steps_str = ", ".join(steps)
        if expose_steps:
            return (
                "Before answering, work through this step-by-step: "
                f"1) {steps[0]} 2) {steps[1]} 3) {steps[2]} 4) {steps[3]} 5) {steps[4]}. "
                "Present your response with brief sections for each step, then a concise final answer."
            )
        else:
            return (
                "Before answering, think through these steps internally: "
                f"{steps_str}. Do not reveal these steps; provide only the concise final answer unless explicitly asked."
            )

    async def chat(self, message, include_vision=False, include_voice=False, use_reasoning: Optional[bool] = None, show_reasoning: Optional[bool] = None, reasoning_preset: Optional[str] = None):
        """Main chat function with Dolphin-Mistral"""
        
        # Handle voice input if requested
        if include_voice and not message:
            voice_text, status = self.listen_for_voice()
            if voice_text:
                message = voice_text
            else:
                return f"🎤 {status}"
        
        # Handle vision input if requested
        vision_context = ""
        if include_vision:
            frame, status = self.capture_image()
            if frame is not None:
                analysis = self.analyze_image(frame)
                vision_context = f"\n\nVision Analysis: {analysis}"
        
        # Prepare conversation
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history (keep last 10 exchanges)
        messages.extend(self.conversation[-20:])
        
        # Auto web fetch: include text from any URLs in the message
        tool_context = ""
        if self.auto_fetch_urls and message:
            urls = self.URL_REGEX.findall(message)
            if urls:
                fetched_blobs = []
                for url in urls[:2]:  # limit
                    fetched = self.fetch_url_text(url)
                    domain = urlparse(url).netloc
                    fetched_blobs.append(f"\n[Source: {domain}]\n{fetched}\n")
                if fetched_blobs:
                    tool_context += "\n\nWEB:\n" + ("\n".join(fetched_blobs))

        # Determine reasoning behavior
        reasoning_enabled = self.structured_reasoning if use_reasoning is None else use_reasoning
        expose_steps = getattr(self, "expose_reasoning", True) if show_reasoning is None else show_reasoning
        preset = self.reasoning_preset if reasoning_preset is None else reasoning_preset

        # Build final user content
        if reasoning_enabled:
            directive = self._build_reasoning_directive(preset, expose_steps)
            base_question = (message or "").strip() + vision_context + tool_context
            full_message = f"{directive}\n\nNow answer: {base_question}".strip()
        else:
            full_message = (message or "") + vision_context + tool_context

        messages.append({"role": "user", "content": full_message})
        
        try:
            # Call Dolphin-Mistral
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            ai_response = response.choices[0].message.content
            
            # Update conversation history
            self.conversation.append({"role": "user", "content": message})
            self.conversation.append({"role": "assistant", "content": ai_response})
            self._save_memory()
            
            return ai_response
            
        except Exception as e:
            return f"❌ Error: {e}"
    
    def generate_code(self, description, language="python"):
        """Generate code using Dolphin-Mistral"""
        prompt = f"""Generate complete, working {language} code for: {description}

Requirements:
- Provide complete, functional code
- Include all necessary imports
- Add error handling where appropriate  
- Make it production-ready
- No explanations, just code

Code:"""
        
        return asyncio.run(self.chat(prompt, use_reasoning=False, show_reasoning=False))

def create_web_interface():
    """Streamlit web interface"""
    st.set_page_config(
        page_title="🦇 BATDAN_BATCOMPUTER",
        page_icon="🦇",
        layout="wide"
    )
    
    # Custom CSS for dark theme
    st.markdown("""
    <style>
    .main-header {
        color: #00ff00;
        text-align: center;
        font-family: 'Courier New', monospace;
        text-shadow: 0 0 10px #00ff00;
    }
    .stChat {
        background-color: #1e1e1e;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🦇 BATDAN_BATCOMPUTER</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #888;">Dolphin-Mistral • Vision • Voice • Unrestricted</p>', unsafe_allow_html=True)
    
    # Initialize BATCOMPUTER
    if 'batcomputer' not in st.session_state:
        st.session_state.batcomputer = SimpleBatComputer()
        if st.session_state.batcomputer.setup_dolphin_mistral():
            st.success("🐬 Dolphin-Mistral initialized successfully!")
        else:
            st.error("❌ Failed to initialize AI model")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # System status
        st.subheader("📊 System Status")
        camera_status = "✅ Active" if st.session_state.batcomputer.camera_active else "❌ Inactive"
        voice_status = "✅ Active" if st.session_state.batcomputer.voice_active else "❌ Inactive"
        
        st.write(f"📹 Camera: {camera_status}")
        st.write(f"🎤 Voice: {voice_status}")
        st.write(f"🤖 Model: {st.session_state.batcomputer.model_name}")
        
        # Input options
        st.subheader("🔧 Input Options")
        use_voice = st.checkbox("🎤 Enable Voice Input")
        use_vision = st.checkbox("📹 Include Vision Analysis")
        auto_web = st.checkbox("🌐 Auto-fetch URLs in prompts")
        show_reasoning = st.checkbox("🧠 Show structured reasoning steps", value=True)
        reasoning_preset = st.selectbox(
            "Reasoning preset",
            ["default", "creative", "analysis", "problem_solving"],
            index=0
        )

        # File access scope
        st.subheader("📁 File Access")
        default_root = getattr(st.session_state.batcomputer, "allowed_root", str(Path.cwd()))
        new_root = st.text_input("Allowed root folder", value=default_root)
        st.session_state.batcomputer.allowed_root = new_root
        st.session_state.batcomputer.auto_fetch_urls = auto_web
        st.session_state.batcomputer.expose_reasoning = show_reasoning
        st.session_state.batcomputer.reasoning_preset = reasoning_preset
        
        # Voice test
        if st.button("🎙️ Voice Test"):
            text, status = st.session_state.batcomputer.listen_for_voice()
            if text:
                st.success(f"Heard: {text}")
                st.session_state.voice_input = text
            else:
                st.error(f"Voice: {status}")
        
        # Vision test
        if st.button("📸 Vision Test"):
            frame, status = st.session_state.batcomputer.capture_image()
            if frame is not None:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=200)
                analysis = st.session_state.batcomputer.analyze_image(frame)
                st.code(analysis, language="json")
            else:
                st.error(f"Camera: {status}")
        
        # Memory controls
        st.subheader("🧠 Memory")
        if st.button("🗑️ Clear Conversation"):
            st.session_state.batcomputer.conversation = []
            if 'chat_history' in st.session_state:
                st.session_state.chat_history = []
            # also clear persisted memory
            try:
                st.session_state.batcomputer.memory_path.unlink(missing_ok=True)
            except Exception:
                pass
            st.rerun()
    
    # Main chat interface
    st.header("💬 Chat with BATCOMPUTER")
    
    # Display chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(f"🦇 **BATCOMPUTER**: {message['content']}")
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message to BATCOMPUTER..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get BATCOMPUTER response
        with st.chat_message("assistant"):
            with st.spinner("🦇 BATCOMPUTER processing..."):
                response = asyncio.run(
                    st.session_state.batcomputer.chat(
                        prompt,
                        include_vision=use_vision,
                        include_voice=False,
                        use_reasoning=True,
                        show_reasoning=show_reasoning,
                        reasoning_preset=reasoning_preset,
                    )
                )
                st.markdown(f"🦇 **BATCOMPUTER**: {response}")
        
        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Voice input from sidebar
    if 'voice_input' in st.session_state and st.session_state.voice_input:
        voice_prompt = st.session_state.voice_input
        st.session_state.voice_input = ""  # Clear it
        
        # Process voice input same as text
        st.session_state.chat_history.append({"role": "user", "content": f"🎤 {voice_prompt}"})
        
        with st.chat_message("user"):
            st.markdown(f"🎤 {voice_prompt}")
        
        with st.chat_message("assistant"):
            with st.spinner("🦇 BATCOMPUTER processing voice..."):
                response = asyncio.run(
                    st.session_state.batcomputer.chat(
                        voice_prompt,
                        include_vision=use_vision,
                        use_reasoning=True,
                        show_reasoning=show_reasoning,
                        reasoning_preset=reasoning_preset,
                    )
                )
                st.markdown(f"🦇 **BATCOMPUTER**: {response}")
                
                # Speak response if voice is enabled
                if use_voice:
                    st.session_state.batcomputer.speak(response[:200])  # Limit speech length
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Code generation section
    st.header("💻 Code Generation")
    col1, col2 = st.columns(2)
    
    with col1:
        code_request = st.text_area("Describe what code you need:", height=100)
        code_language = st.selectbox(
            "Language:", 
            ["python", "javascript", "html", "css", "powershell", "batch", "c++", "java", "rust", "go"]
        )
    
    with col2:
        if st.button("🚀 Generate Code") and code_request:
            with st.spinner("Generating code..."):
                code = st.session_state.batcomputer.generate_code(code_request, code_language)
                st.code(code, language=code_language)
                
                # Option to save code
                if st.download_button(
                    label="💾 Download Code",
                    data=code,
                    file_name=f"generated_code.{code_language}",
                    mime="text/plain"
                ):
                    st.success("Code saved!")

# Terminal interface function
def run_terminal():
    """Run BATCOMPUTER in terminal mode"""
    print("🦇 BATDAN_BATCOMPUTER Terminal Interface")
    print("=====================================")
    
    batcomputer = SimpleBatComputer()
    if not batcomputer.setup_dolphin_mistral():
        print("❌ Failed to setup AI model")
        return
    
    print("\n🦇 BATCOMPUTER ready! Type 'quit' to exit.")
    print("Commands: 'voice' for voice input, 'vision' to include camera")
    
    while True:
        try:
            user_input = input("\n🦇 BATDAN> ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 BATCOMPUTER shutting down...")
                break
            elif user_input.lower() == 'voice':
                print("🎤 Listening for voice input...")
                response = asyncio.run(batcomputer.chat("", include_voice=True))
                print(f"\n🦇 BATCOMPUTER: {response}")
            elif user_input.lower() == 'vision':
                print("📹 Including vision analysis...")
                response = asyncio.run(batcomputer.chat("Analyze what you can see", include_vision=True))
                print(f"\n🦇 BATCOMPUTER: {response}")
            elif user_input:
                response = asyncio.run(batcomputer.chat(user_input))
                print(f"\n🦇 BATCOMPUTER: {response}")
                
        except KeyboardInterrupt:
            print("\n👋 BATCOMPUTER shutting down...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "terminal":
        run_terminal()
    else:
        create_web_interface()