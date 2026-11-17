# All settings come from the .env file.
# The booking facts below are the ONLY facts the agent is allowed to use.

import os
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()


def env(name, default=""):
    # "KEY=" with nothing after it counts as not set, so the default is used.
    return os.getenv(name) or default


# Telegram user account (from https://my.telegram.org)
TG_API_ID = int(env("TG_API_ID", "0"))
TG_API_HASH = env("TG_API_HASH", "")
TG_SESSION = env("TG_SESSION", "agent")
TARGET = env("TARGET", "")          # manager's @username
OWNER_CHAT = env("OWNER_CHAT", "me")  # where the agent asks me; "me" = Saved Messages

# LLM (any OpenAI-compatible API)
LLM_BASE_URL = env("LLM_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
LLM_API_KEY = env("LLM_API_KEY", "")
LLM_MODEL = env("LLM_MODEL", "qwen3.8-flash")

# Voice services on the GPU computer (see voice.py). Empty = voice switched off.
VOICE_TTS_URL = env("VOICE_TTS_URL", "").rstrip("/")
VOICE_TTS_KEY = env("VOICE_TTS_KEY", "")
VOICE_NAME = env("VOICE_NAME", "Fleurs uk F 10")  # a Zonos voice from GET /voices/zonos
VOICE_STT_URL = env("VOICE_STT_URL", "").rstrip("/")

# Booking facts. Empty means "unknown": the agent must ask, not guess.
GUEST_NAME = env("GUEST_NAME", "")
GUEST_PHONE = env("GUEST_PHONE", "")
PARTY_SIZE = int(env("PARTY_SIZE", "2"))
BOOKING_TIME = env("BOOKING_TIME", "17:00")
RESTAURANT_NAME = env("RESTAURANT_NAME", "")
TIMEZONE = ZoneInfo(env("BOOKING_TZ", "Europe/London"))


def london_now():
    # The restaurant is in London, so "17:00 today" means London time
    # (17:00 in London is 19:00 in Kyiv in winter).
    return datetime.now(TIMEZONE)


def today_text():
    # e.g. "Monday, 16 November 2026" (date in London, not in Kyiv).
    # Late in the evening the dates can differ, so we never use the local date.
    now = london_now()
    return f"{now:%A}, {now.day} {now:%B %Y}"
