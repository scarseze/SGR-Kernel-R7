import os
from openai import OpenAI

class VoiceService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        # Debug Key (print first 10 chars)
        masked_key = (self.api_key[:10] + "...") if self.api_key else "None"
        print(f"DEBUG: VoiceService loaded GROQ key: {masked_key}")
        
        if not self.api_key:
            print("Warning: GROQ_API_KEY not found. Voice transcription will fail.")

        # Check for Proxy
        proxy_url = os.getenv("PROXY_URL") # e.g. http://127.0.0.1:8080
        http_client = None
        if proxy_url:
            import httpx
            print(f"DEBUG: VoiceService using proxy: {proxy_url}")
            http_client = httpx.Client(proxy=proxy_url)
            
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key,
            http_client=http_client
        )
        self.model = "whisper-large-v3"

    def transcribe(self, audio_file_path: str) -> str:
        """
        Transcribes an audio file using Groq's Whisper model.
        Returns the text.
        """
        if not self.api_key:
            return "Error: No GROQ_API_KEY provided."
            
        try:
            if not os.path.exists(audio_file_path):
                return f"Error: Audio file not found at {audio_file_path}"
                
            with open(audio_file_path, "rb") as file:
                print(f"DEBUG: Sending audio to Groq ({self.client.base_url})...")
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(audio_file_path), file.read()),
                    model=self.model,
                    response_format="text" # or json
                )
            
            # OpenAI client usually returns an object if format is json, or str if text
            # Groq implementation might vary, but 'text' format ensures string
            return str(transcription)
            
        except Exception as e:
            return f"Transcription Failed: {str(e)}"
