# Parlor

On-device, real-time multimodal AI. Have natural voice and vision conversations with an AI that runs entirely on your machine — nothing leaves your device.

Parlor pairs [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) (2.3B effective parameters, native audio + vision) with [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS for fluid, spoken conversations. You talk, show your camera, and it talks back — all locally.

https://github.com/user-attachments/assets/placeholder

## How it works

```
Browser (mic + camera)
    │
    │  WebSocket (audio PCM + JPEG frames)
    ▼
FastAPI server
    ├── Gemma 4 E2B via LiteRT-LM (GPU)  →  understands speech + vision
    └── Kokoro TTS (MLX on Mac, ONNX on Linux)  →  speaks back
    │
    │  WebSocket (streamed audio chunks)
    ▼
Browser (playback + transcript)
```

- **Voice Activity Detection** runs in the browser ([Silero VAD](https://github.com/ricky0123/vad)) — hands-free, push-to-talk free
- **Barge-in** — interrupt the AI mid-sentence by speaking
- **Sentence-level TTS streaming** — audio starts playing before the full response is generated

## Requirements

- **Python 3.12+**
- **macOS** with Apple Silicon (M1/M2/M3/M4) — uses Metal via WebGPU for GPU inference
- **Linux** with a supported GPU — uses ONNX for TTS (CPU)
- ~3 GB free RAM for the Gemma 4 E2B model

## Setup

### 1. Install dependencies

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

cd src
uv sync
```

### 2. Download the model

```bash
# Install the LiteRT-LM CLI
uv tool install litert-lm

# Download Gemma 4 E2B (~2.6 GB)
litert-lm download litert-community/gemma-4-E2B-it-litert-lm
```

This downloads `gemma-4-E2B-it.litertlm`. Note the path — you'll need it next.

> **Linux only:** You also need the TTS model files. On macOS these are downloaded automatically.
> ```bash
> cd src
> curl -LO https://github.com/hexgrad/Kokoro-82M/releases/download/v1.0/kokoro-v1.0.onnx
> curl -LO https://github.com/hexgrad/Kokoro-82M/releases/download/v1.0/voices-v1.0.bin
> ```

### 3. Run

```bash
cd src
MODEL_PATH=/path/to/gemma-4-E2B-it.litertlm uv run python server.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser. Grant camera and microphone access when prompted — start talking.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | (required) | Path to `gemma-4-E2B-it.litertlm` |
| `PORT` | `8000` | Server port |

See [`.env.example`](.env.example) for a template.

## Performance (Apple M3 Pro)

| Stage | Time |
|-------|------|
| Speech + vision understanding | ~1.8-2.2s |
| Response generation (~25 tokens) | ~0.3s |
| Text-to-speech (1-3 sentences) | ~0.3-0.7s |
| **Total end-to-end** | **~2.5-3.0s** |

Decode speed: ~83 tokens/sec (GPU).

## Project structure

```
src/
├── server.py          # FastAPI WebSocket server + Gemma 4 inference
├── tts.py             # Platform-aware TTS (MLX on Mac, ONNX on Linux)
├── index.html         # Frontend UI (VAD, camera, audio playback)
└── pyproject.toml     # Dependencies
benchmarks/
├── bench.py           # End-to-end WebSocket benchmark
└── benchmark_tts.py   # TTS backend comparison
```

## Acknowledgments

- [Gemma 4](https://ai.google.dev/gemma) by Google DeepMind
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) by Google AI Edge
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS by Hexgrad
- [Silero VAD](https://github.com/snakers4/silero-vad) for browser voice activity detection

## License

Apache 2.0 — see [LICENSE](LICENSE).
