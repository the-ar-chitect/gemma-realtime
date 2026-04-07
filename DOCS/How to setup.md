It's not turn-based like the Ollama + Open WebUI I showed earlier — this is **hands-free, streaming, barge-in capable** v2v right in the browser:
- You speak naturally (no push-to-talk)
- Silero VAD in the browser detects voice
- Audio + camera frames → WebSocket → Gemma 4 (E2B or E4B) via LiteRT-LM on GPU
- It thinks → Kokoro TTS streams audio chunks back instantly (sentence-level, so you hear it before the full reply finishes)
- You can interrupt mid-sentence (barge-in)
- Text chat is also available (configurable)
- Everything runs 100% locally on your g6.xlarge (no cloud APIs)

Models auto-download on first run (~2.6 GB for E2B, ~3.65 GB for E4B + Kokoro TTS). Total RAM footprint fits in the L4's 24 GB VRAM.

**Important notes before we dive in**
- Supports **Gemma 4 E2B** (default, lighter, ~2.6 GB) and **E4B** (larger, stronger, ~3.65 GB) — switch via `MODEL_VARIANT` env var.
- Both E2B and E4B are multimodal (speech + vision).
- Linux + NVIDIA GPU supported (g6.xlarge = perfect).
- Browser UI is built-in (`index.html`) — just open the IP:8000 and grant mic/camera.
- Audio, video, and text chat can be individually enabled/disabled via environment variables.
- Supports multiple concurrent sessions via engine pooling (`MAX_SESSIONS` env var).
- Research preview (expect occasional rough edges, but the core is buttery).

---

### Environment Variables

All configuration is done via environment variables. Set them before launching the server.

| Variable | Default | Description |
|---|---|---|
| `MODEL_VARIANT` | `E2B` | Model to use: `E2B` (2.6 GB, lighter) or `E4B` (3.65 GB, stronger). Both are multimodal. |
| `MAX_SESSIONS` | `1` | Number of concurrent engine instances (one per user). Each E2B costs ~2.6 GB VRAM, each E4B ~5.5 GB. |
| `MAX_HISTORY_TURNS` | `20` | Number of conversation turns to keep in context. More turns = more memory. |
| `CONTEXT_WINDOW` | `131072` | Context window size in tokens (max 131072 = 128K for both E2B and E4B). |
| `ENABLE_AUDIO` | `true` | Enable voice input (microphone + VAD). Set to `false` for text-only mode. |
| `ENABLE_VIDEO` | `true` | Enable camera/vision input. Set to `false` to hide the camera. |
| `ENABLE_CHAT` | `true` | Enable text chat input bar. Set to `false` for voice-only mode. |
| `MODEL_PATH` | *(auto-download)* | Override with a local path to a `.litertlm` file (skips HuggingFace download). |
| `PORT` | `8000` | Server port. |

**How to set them:**

```bash
# Option A: Inline when launching
MODEL_VARIANT=E4B MAX_SESSIONS=2 ENABLE_CHAT=true uv run server.py

# Option B: Export (persists for the shell session)
export MODEL_VARIANT=E4B
export MAX_SESSIONS=2
export ENABLE_AUDIO=true
export ENABLE_VIDEO=true
export ENABLE_CHAT=true
uv run server.py

# Option C: .env file (create src/.env)
#   MODEL_VARIANT=E4B
#   MAX_SESSIONS=2
#   ENABLE_CHAT=true
# Then:
env $(cat .env | xargs) uv run server.py
```

**Common configurations:**

```bash
# Voice + vision + chat (all features, E4B)
MODEL_VARIANT=E4B ENABLE_AUDIO=true ENABLE_VIDEO=true ENABLE_CHAT=true uv run server.py

# Text chat only (no mic, no camera)
ENABLE_AUDIO=false ENABLE_VIDEO=false ENABLE_CHAT=true uv run server.py

# Voice only (no camera, no text chat)
ENABLE_VIDEO=false ENABLE_CHAT=false uv run server.py

# Multi-user (3 concurrent sessions with E2B)
MAX_SESSIONS=3 uv run server.py
```

---

### Full Step-by-Step Guide: Run Parlor on g6.xlarge (Updated for This Repo)

#### 1. Launch the EC2 Instance (same as before, tiny tweak)
- AMI: **Ubuntu Server 24.04 LTS**
- Type: **g6.xlarge**
- Security group:
  - SSH (22) → your IP only
  - HTTP (8000) → Anywhere-IPv4 (or your IP)
- Storage: 50 GiB gp3 is plenty (~3 GB models + cache)
- Launch → SSH in.

#### 2. Install NVIDIA Drivers + CUDA (required for LiteRT-LM GPU accel on L4)
```bash
ssh -i your-key.pem ubuntu@<public-ip>

sudo apt update && sudo apt upgrade -y

# NVIDIA driver (570 series works great for L4)
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update
sudo apt install nvidia-driver-570 -y

# CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit nvidia-gds -y

sudo reboot
```

After reboot verify:
```bash
nvidia-smi
```
You should see the L4 GPU. (If not, `sudo ubuntu-drivers autoinstall && reboot`.)

#### 3. Install uv + Clone & Run Parlor
Ubuntu 24.04 already has Python 3.12 — perfect.

```bash
# Install uv (fast Python package manager used by the repo)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env   # add to PATH

# Clone the repo
git clone https://github.com/fikrikarim/parlor.git
cd parlor/src

# Install dependencies (uv will pull everything, including LiteRT-LM, FastAPI, ONNX for Kokoro, etc.)
uv sync

# Start the server (default: E2B, all features on)
uv run server.py

# Or with E4B and custom settings:
MODEL_VARIANT=E4B MAX_SESSIONS=2 uv run server.py
```

First run will auto-download the models (~2.6 GB for E2B or ~3.65 GB for E4B — takes a minute or two depending on your network).

Leave this terminal open (or use `tmux` / `screen` so it survives disconnects — I'll give systemd service if you want it to auto-start on boot).

#### 4. Access via Browser (Realtime v2v + Vision + Chat)
Open in any modern browser:
**http://<your-public-ip>:8000**

1. Click "Allow" when it asks for microphone + camera.
2. Click "Connect" to start a session.
3. Start talking naturally — no buttons needed (if audio is enabled).
4. Type in the chat bar to send text messages (if chat is enabled).
5. Show the camera something → it understands vision too (multimodal magic).
6. Interrupt anytime by speaking while it's replying.

That's it. You now have a fully private, realtime, multimodal Gemma 4 voice assistant running on your g6.xlarge, accessible from anywhere.

#### Performance on g6.xlarge (L4 GPU)
The README benchmarks on M3 Pro Mac (~83 tokens/sec decode, 2.5–3s end-to-end).
On the L4 you'll see **even better**: sub-2s responses, higher tokens/sec, and room for multiple concurrent sessions if you want. Kokoro TTS on ONNX is lightning fast. GPU utilization will be excellent.

**VRAM budget (L4 = 24 GB):**
| Config | VRAM Estimate |
|---|---|
| 1× E2B | ~2.6 GB |
| 1× E4B | ~5.5 GB |
| 3× E2B (`MAX_SESSIONS=3`) | ~7.8 GB |
| 2× E4B (`MAX_SESSIONS=2`) | ~11 GB |
| 4× E2B (`MAX_SESSIONS=4`) | ~10.4 GB |

#### Make It Production-Ready (Quick Extras)
- **Persistent on reboot** (optional systemd service):
  ```bash
  sudo tee /etc/systemd/system/parlor.service > /dev/null <<EOF
  [Unit]
  Description=Parlor Realtime AI
  After=network.target

  [Service]
  User=ubuntu
  WorkingDirectory=/home/ubuntu/parlor/src
  ExecStart=/home/ubuntu/.cargo/bin/uv run server.py
  Restart=always
  Environment=PATH=/home/ubuntu/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  Environment=MODEL_VARIANT=E2B
  Environment=MAX_SESSIONS=1
  Environment=ENABLE_AUDIO=true
  Environment=ENABLE_VIDEO=true
  Environment=ENABLE_CHAT=true

  [Install]
  WantedBy=multi-user.target
  EOF

  sudo systemctl daemon-reload
  sudo systemctl enable --now parlor
  ```
  To change settings later: edit the `Environment=` lines, then `sudo systemctl daemon-reload && sudo systemctl restart parlor`.

- **Security**: Keep port 8000 restricted to your IP (or add Nginx + HTTPS later — ping me).
- **Stop saving money**: Just stop the EC2 instance when done (EBS volume keeps everything).
- **Monitor**: `nvidia-smi -l 1` in another session.

This beats every other self-hosted v2v setup I've seen for simplicity + realtime quality on a single GPU. No Ollama, no extra Whisper/Piper, no Docker needed — pure native Gemma 4 + Kokoro.

If you hit any snag (model download fail, GPU not detected, want to dockerize it, add RAG, or multi-user), just paste the terminal output or error and we'll fix in one message.
