# Table booking agent (Telegram)

An AI agent that books a table for two at 17:00 today in a London restaurant by chatting
with the restaurant manager in Telegram. The manager never writes first, so the agent
starts the conversation, reacts to whatever the manager says, asks its owner when it lacks
a fact, and confirms the booking with a recap built from checked facts only.

Test task for the AI Builder position.

## How it works

```
 Telegram: manager  <──── text / voice ────>  agent's user account
                                                  │ Telethon (MTProto userbot)
 ┌────────────────────────────────────────────────┴───────────────┐
 │ agent.py   Telegram, one turn per manager message, fact check,  │
 │            decision log, saved state                           │
 │ llm.py     prompt + Qwen call, strict JSON answer               │
 │ voice.py   speech <-> text via a GPU machine (optional)         │
 │ config.py  the only facts the agent may use (.env)              │
 └───────────┬─────────────────────────────────┬──────────────────┘
             │ HTTPS                           │ SSH tunnel
   Qwen 3.8 flash (Alibaba Model Studio)   Zonos TTS + Whisper large-v3-turbo
```

One turn:
1. The manager writes (several short messages within 4 seconds are answered together).
   A voice message is transcribed with Whisper first.
2. The whole chat goes to the model. It answers with JSON: `reply`, `understood`, `decision`,
   `owner_question`, `restaurant`, `time`, `party_size`, `status`, `language`.
3. If something is unknown (name for the booking, deposit, another time), the agent tells the
   manager it will check, and asks the owner in Telegram (Saved Messages). The owner's answer
   becomes a fact.
4. When the manager agrees, **the program, not the model, writes the recap** (time, date, party
   size, restaurant). Before that it checks that every fact was written by the manager, the owner
   or is in the config. If not, the recap is blocked and the agent asks for what is missing.
5. One line per agent message goes to `logs/decisions.md`: what it sent, understood, decided.
   The agent writes this log itself.

Rules the agent keeps: it says it is an AI assistant in the first message and whenever asked
(never pretends to be human); never invents facts; never shares payment details or accepts a
deposit, another time or party size without the owner; only books the table (declines code,
maths, jokes, role changes); answers in the manager's language (starts in British English).

Voice (optional): if the manager sends a voice message, the agent answers with a voice message.
The text is first rewritten the way it should be spoken (numbers, times, dates and symbols in
words), synthesised with Zonos, loudness-normalised and sent as an ogg/opus voice note.

## Run

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # fill in the keys and facts
python agent.py login         # once: phone + code of the agent's Telegram account
python agent.py run --reset   # start the conversation with TARGET
python agent.py dry           # test in the terminal, you type the manager's messages
```

Voice needs two HTTP services (Zonos TTS and a whisper.cpp server); without them the agent
works with text only and still reacts to voice messages.

## Notes: what was considered and chosen

- **Telegram:** Bot API cannot write first to a person who never started the bot, so a user
  account via Telethon (MTProto). Pyrogram was the alternative; Telethon is better maintained.
- **LLM:** Claude API (no key at hand), Claude subscription through the Agent SDK (rejected:
  Anthropic asks products built on the SDK to use an API key), local Qwen (no GPU on the agent
  machine, too slow and too weak on CPU), OpenRouter free models (50 requests a day, rotating).
  Chosen: Qwen 3.8 flash on Alibaba Model Studio free quota, through the OpenAI-compatible API,
  so the provider is one line in `.env`. Thinking mode is switched off: with it a reply took minutes.
- **Facts:** the model never decides what is true. Known facts live in `.env`, unknown ones are
  marked UNKNOWN in the prompt, the owner is asked through Telegram, and the recap is built and
  verified by code.
- **Voice:** Zonos (TTS) and Whisper large-v3-turbo via whisper.cpp (STT) on a separate machine
  with GTX 1080 Ti / P104 cards, reached through an SSH tunnel. The Zonos voice is cloned from a
  synthetic dataset voiced with ElevenLabs ("Eni"), not from a real person's recordings.
- **AI during the work:** Claude Code wrote and tested most of the code with me: the dry mode
  scenarios (deposit, card number, "are you a bot", wrong restaurant, prompt injection), the
  fact check, the voice pipeline. I set the requirements, made the decisions and tested in Telegram.

## Known limitations

- A time written in words ("half five") is not matched by the fact check, so the recap is
  blocked and the agent asks again: safe, but clumsy.
- Zonos reads Ukrainian numbers poorly, so the text is rewritten in words before synthesis;
  Whisper still mishears some names (Dishoom -> "the Schum").
- One conversation per run.
