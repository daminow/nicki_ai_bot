import tempfile
import subprocess
from pathlib import Path
from openai import OpenAI
from .config import settings

openai_client = OpenAI(api_key=settings.novita_api_key)


def ogg_to_wav(ogg_path: Path) -> Path:
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    subprocess.run(
        ["ffmpeg", "-i", str(ogg_path), "-ac", "1", "-ar", "16000", wav_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return Path(wav_path)


async def speech_to_text(ogg_path: Path) -> str:
    wav_path = ogg_to_wav(ogg_path)
    with open(wav_path, "rb") as f:
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1", file=f
        )
    return transcription.text