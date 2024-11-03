import asyncio
import websockets
import whisper
import tempfile
from pydub import AudioSegment
import os

# Load the Whisper model
model = whisper.load_model("base")

async def transcribe_audio(websocket, path):
    print("Client connected")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as audio_file:
        audio_file_path = audio_file.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
        wav_file_path = wav_file.name

    try:
        while True:
            audio_chunk = await websocket.recv()
            
            with open(audio_file_path, "ab") as f:
                f.write(audio_chunk)

            # Convert webm to wav
            audio = AudioSegment.from_file(audio_file_path, format="webm")
            audio.export(wav_file_path, format="wav")

            # Transcribe
            result = model.transcribe(wav_file_path, fp16=False)
            transcript = result.get("text", "")

            # Send transcription to the client
            await websocket.send(transcript)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        os.remove(audio_file_path)
        os.remove(wav_file_path)

async def main():
    async with websockets.serve(transcribe_audio, "localhost", 8080):
        print("WebSocket server started on ws://localhost:8080")
        await asyncio.Future()

asyncio.run(main())
