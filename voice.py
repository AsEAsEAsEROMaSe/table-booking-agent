# Voice messages (optional).
# The models do not run on this machine (no GPU). They run on a separate GPU computer,
# reached through an SSH tunnel (see README):
#   TTS: Zonos, through the NOTQA voice service:  POST /tts {engine, text, voice, lang} -> wav
#   STT: Whisper large-v3-turbo (whisper.cpp server): POST /inference (wav file) -> {"text"}
# If VOICE_TTS_URL / VOICE_STT_URL are empty in .env, voice is switched off
# and the agent works with text only.

import os
import subprocess
import tempfile

import requests

import config


def can_listen():
    return bool(config.VOICE_STT_URL)


def can_speak():
    return bool(config.VOICE_TTS_URL)


def warm_up():
    """Zonos is unloaded after some idle time and the first phrase then takes ~30 s.
    Loading it in advance keeps the first voice reply to the manager quick."""
    try:
        requests.post(f"{config.VOICE_TTS_URL}/engines/zonos/start",
                      headers={"X-TTS-Key": config.VOICE_TTS_KEY}, timeout=120)
    except Exception as e:
        print("Voice warm-up failed (voice replies may be slow):", e)


def convert(src, dst, *ffmpeg_args):
    """Small wrapper around ffmpeg."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, *ffmpeg_args, dst], check=True)


def speech_to_text(audio_path):
    """Turn the manager's voice message into text with Whisper."""
    # Telegram voice is ogg/opus; whisper.cpp wants 16 kHz mono wav.
    wav_path = audio_path + ".wav"
    convert(audio_path, wav_path, "-ar", "16000", "-ac", "1")
    with open(wav_path, "rb") as f:
        response = requests.post(f"{config.VOICE_STT_URL}/inference",
                                 files={"file": f},
                                 data={"response_format": "json", "language": "auto"},
                                 timeout=120)
    response.raise_for_status()
    return response.json()["text"].strip()


def text_to_voice(text, language="en"):
    """Turn the agent's reply into a Telegram voice note. Returns the path of an .ogg file."""
    response = requests.post(f"{config.VOICE_TTS_URL}/tts",
                             headers={"X-TTS-Key": config.VOICE_TTS_KEY},
                             json={"engine": "zonos", "text": text,
                                   "voice": config.VOICE_NAME, "lang": language},
                             timeout=300)
    response.raise_for_status()

    # Telegram shows a file as a voice note only if it is ogg/opus.
    folder = tempfile.mkdtemp()
    wav_path = os.path.join(folder, "reply.wav")
    ogg_path = os.path.join(folder, "reply.ogg")
    with open(wav_path, "wb") as f:
        f.write(response.content)
    # loudnorm evens out the volume (podcast level, -16 LUFS), so the voice is clearly heard
    # on a phone; the short silence at the start keeps the first word from being cut.
    convert(wav_path, ogg_path,
            "-af", "adelay=150:all=1,loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar", "48000", "-c:a", "libopus", "-b:a", "48k", "-ac", "1")
    return ogg_path
