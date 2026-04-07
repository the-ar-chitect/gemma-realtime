"""Parlor — on-device, real-time multimodal AI (voice + vision)."""

import asyncio
import base64
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import litert_lm
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import tts

# Model variant: "E2B" (2.6 GB, lighter) or "E4B" (3.65 GB, stronger).
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "E2B").upper()

# Max conversation turns to keep (controls effective context usage).
# Gemma 4 E2B/E4B supports 128K tokens; each turn with image+audio is ~300-500 tokens.
# Default 20 turns ≈ 6K-10K tokens of history.
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "20"))

# Number of concurrent sessions (each needs its own engine instance, ~2.6-3.7 GB VRAM).
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "1"))

# Context window (tokens). Gemma 4 E2B/E4B supports up to 128K.
CONTEXT_WINDOW = int(os.environ.get("CONTEXT_WINDOW", "131072"))

# Feature toggles — disable individual input modalities from the server.
ENABLE_AUDIO = os.environ.get("ENABLE_AUDIO", "true").lower() in ("true", "1", "yes")
ENABLE_VIDEO = os.environ.get("ENABLE_VIDEO", "true").lower() in ("true", "1", "yes")
ENABLE_CHAT  = os.environ.get("ENABLE_CHAT", "true").lower() in ("true", "1", "yes")

_MODEL_VARIANTS = {
    "E2B": {
        "repo": "litert-community/gemma-4-E2B-it-litert-lm",
        "filename": "gemma-4-E2B-it.litertlm",
    },
    "E4B": {
        "repo": "litert-community/gemma-4-E4B-it-litert-lm",
        "filename": "gemma-4-E4B-it.litertlm",
    },
}

if MODEL_VARIANT not in _MODEL_VARIANTS:
    raise ValueError(f"Unknown MODEL_VARIANT={MODEL_VARIANT!r}. Use 'E2B' or 'E4B'.")

HF_REPO = _MODEL_VARIANTS[MODEL_VARIANT]["repo"]
HF_FILENAME = _MODEL_VARIANTS[MODEL_VARIANT]["filename"]


def resolve_model_path() -> str:
    path = os.environ.get("MODEL_PATH", "")
    if path:
        return path
    from huggingface_hub import hf_hub_download
    print(f"Downloading {HF_REPO}/{HF_FILENAME} (first run only)...")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


MODEL_PATH = resolve_model_path()

def _build_system_prompt() -> str:
    """Build system prompt based on enabled modalities."""
    parts = ["You are a friendly, conversational AI assistant."]
    if ENABLE_AUDIO and ENABLE_VIDEO:
        parts.append("The user is talking to you through a microphone and showing you their camera.")
    elif ENABLE_AUDIO:
        parts.append("The user is talking to you through a microphone.")
    elif ENABLE_VIDEO:
        parts.append("The user is showing you their camera and chatting via text.")
    else:
        parts.append("The user is chatting with you via text.")
    parts.append(
        "You MUST always use the respond_to_user tool to reply. "
        "First transcribe exactly what the user said (or restate their text), then write your response."
    )
    parts.append(
        "You also have test tools numbered 1 through 10 available. "
        "When the user asks you to call a specific tool by number, call that tool."
    )
    return " ".join(parts)

SYSTEM_PROMPT = _build_system_prompt()

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

engine_pool: asyncio.Queue = asyncio.Queue()
active_sessions = 0  # track how many engines are currently checked-out
tts_backend = None


def load_models():
    global tts_backend
    for i in range(MAX_SESSIONS):
        print(f"Loading engine {i+1}/{MAX_SESSIONS} from {MODEL_PATH}...")
        eng = litert_lm.Engine(
            MODEL_PATH,
            backend=litert_lm.Backend.GPU,
            vision_backend=litert_lm.Backend.GPU,
            audio_backend=litert_lm.Backend.CPU,
        )
        eng.__enter__()
        engine_pool.put_nowait(eng)
        print(f"Engine {i+1}/{MAX_SESSIONS} loaded.")

    tts_backend = tts.load()


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent), name="static")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.get("/config")
async def get_config():
    return {
        "max_history_turns": MAX_HISTORY_TURNS,
        "max_sessions": MAX_SESSIONS,
        "active_sessions": active_sessions,
        "available_sessions": engine_pool.qsize(),
        "model": HF_FILENAME,
        "model_variant": MODEL_VARIANT,
        "context_window": CONTEXT_WINDOW,
        "enable_audio": ENABLE_AUDIO,
        "enable_video": ENABLE_VIDEO,
        "enable_chat": ENABLE_CHAT,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global active_sessions
    await ws.accept()

    # Try to check out an engine from the pool (non-blocking).
    try:
        eng = engine_pool.get_nowait()
    except asyncio.QueueEmpty:
        await ws.send_text(json.dumps({
            "type": "error",
            "error": f"All {MAX_SESSIONS} session(s) are in use. Try again shortly.",
        }))
        await ws.close(code=1008, reason="Session busy")
        return

    active_sessions += 1
    try:
        await _run_session(ws, eng)
    finally:
        active_sessions -= 1
        engine_pool.put_nowait(eng)


async def _run_session(ws: WebSocket, engine):
    # Per-connection tool state captured via closure
    tool_result = {}
    tool_calls = []  # Track test tool invocations

    def respond_to_user(transcription: str, response: str) -> str:
        """Respond to the user's voice message.

        Args:
            transcription: Exact transcription of what the user said in the audio.
            response: Your conversational response to the user. Keep it to 1-4 short sentences.
        """
        tool_result["transcription"] = transcription
        tool_result["response"] = response
        return "OK"

    # Test tools 1-10: simple tools the LLM can call to verify tool-calling works
    def _make_test_tool(n: int):
        def tool_fn(reason: str) -> str:
            """Placeholder docstring (overridden below)."""
            tool_calls.append({"tool": n, "reason": reason})
            return f"Tool {n} executed successfully."
        tool_fn.__name__ = f"test_tool_{n}"
        tool_fn.__doc__ = (
            f"Test tool {n}. Call this when the user asks you to call tool {n}.\n\n"
            f"Args:\n    reason: Brief explanation of why this tool was called."
        )
        return tool_fn

    test_tools = [_make_test_tool(i) for i in range(1, 11)]

    print(f"Session started (active: {active_sessions}/{MAX_SESSIONS})")
    conversation = engine.create_conversation(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=[respond_to_user] + test_tools,
    )
    conversation.__enter__()
    turn_count = 0

    interrupted = asyncio.Event()
    msg_queue = asyncio.Queue()

    async def receiver():
        """Receive messages from WebSocket and route them."""
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            interrupted.clear()

            content = []
            if msg.get("audio"):
                content.append({"type": "audio", "blob": msg["audio"]})
            if msg.get("image"):
                content.append({"type": "image", "blob": msg["image"]})

            if msg.get("audio") and msg.get("image"):
                content.append({"type": "text", "text": "The user just spoke to you (audio) while showing their camera (image). Respond to what they said, referencing what you see if relevant."})
            elif msg.get("audio"):
                content.append({"type": "text", "text": "The user just spoke to you. Respond to what they said."})
            elif msg.get("image") and msg.get("text"):
                content.append({"type": "text", "text": f"The user typed: {msg['text']}\nThey are also showing their camera. Respond to what they said, referencing what you see if relevant."})
            elif msg.get("image"):
                content.append({"type": "text", "text": "The user is showing you their camera. Describe what you see."})
            else:
                content.append({"type": "text", "text": msg.get("text", "Hello!")})

            # LLM inference
            t0 = time.time()
            tool_result.clear()
            tool_calls.clear()
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: conversation.send_message({"role": "user", "content": content})
            )
            llm_time = time.time() - t0
            turn_count += 1

            # Send test tool call notifications to frontend
            for tc in tool_calls:
                await ws.send_text(json.dumps({
                    "type": "tool_call",
                    "tool": tc["tool"],
                    "reason": tc["reason"],
                }))
                print(f"Tool {tc['tool']} called: {tc['reason']}")

            # Extract response from tool call or fallback to raw text
            if tool_result:
                strip = lambda s: s.replace('<|"|>', "").strip()
                transcription = strip(tool_result.get("transcription", ""))
                text_response = strip(tool_result.get("response", ""))
                print(f"LLM ({llm_time:.2f}s) [tool] heard: {transcription!r} → {text_response}")
            else:
                transcription = None
                text_response = response["content"][0]["text"]
                print(f"LLM ({llm_time:.2f}s) [no tool]: {text_response}")

            if interrupted.is_set():
                print("Interrupted after LLM, skipping response")
                continue

            reply = {
                "type": "text", "text": text_response,
                "llm_time": round(llm_time, 2),
                "turn": turn_count,
                "max_turns": MAX_HISTORY_TURNS,
            }
            if transcription:
                reply["transcription"] = transcription
            if tool_calls:
                reply["tools_called"] = [tc["tool"] for tc in tool_calls]
            await ws.send_text(json.dumps(reply))

            if interrupted.is_set():
                print("Interrupted before TTS, skipping audio")
                continue

            # Streaming TTS: split into sentences and send chunks progressively
            sentences = split_sentences(text_response)
            if not sentences:
                sentences = [text_response]

            tts_start = time.time()

            # Signal start of audio stream
            await ws.send_text(json.dumps({
                "type": "audio_start",
                "sample_rate": tts_backend.sample_rate,
                "sentence_count": len(sentences),
            }))

            for i, sentence in enumerate(sentences):
                if interrupted.is_set():
                    print(f"Interrupted during TTS (sentence {i+1}/{len(sentences)})")
                    break

                # Generate audio for this sentence
                pcm = await asyncio.get_event_loop().run_in_executor(
                    None, lambda s=sentence: tts_backend.generate(s)
                )

                if interrupted.is_set():
                    break

                # Convert to 16-bit PCM and send as base64
                pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                await ws.send_text(json.dumps({
                    "type": "audio_chunk",
                    "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                    "index": i,
                }))

            tts_time = time.time() - tts_start
            print(f"TTS ({tts_time:.2f}s): {len(sentences)} sentences")

            if not interrupted.is_set():
                await ws.send_text(json.dumps({
                    "type": "audio_end",
                    "tts_time": round(tts_time, 2),
                }))

    except WebSocketDisconnect:
        print(f"Client disconnected (active after: {active_sessions - 1}/{MAX_SESSIONS})")
    finally:
        recv_task.cancel()
        conversation.__exit__(None, None, None)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
