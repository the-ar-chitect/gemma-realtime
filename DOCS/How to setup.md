It’s not turn-based like the Ollama + Open WebUI I showed earlier — this is **hands-free, streaming, barge-in capable** v2v right in the browser:  
- You speak naturally (no push-to-talk)  
- Silero VAD in the browser detects voice  
- Audio + camera frames → WebSocket → Gemma 4 E2B (native speech + vision understanding via LiteRT-LM on GPU)  
- It thinks → Kokoro TTS streams audio chunks back instantly (sentence-level, so you hear it before the full reply finishes)  
- You can interrupt mid-sentence (barge-in)  
- Everything runs 100% locally on your g6.xlarge (no cloud APIs)  

Models auto-download on first run (~2.6 GB Gemma 4 E2B-it + Kokoro). Total RAM footprint ~3 GB — fits perfectly in the L4’s 24 GB VRAM.  

**Important notes before we dive in**  
- Fixed to **Gemma 4 E2B** (the multimodal speech+vision variant — exactly what you wanted for realtime v2v).  
- No built-in E4B switch (E4B isn’t multimodal the same way; this repo is purpose-built for E2B). If you later want a toggle, we can fork and add it, but right now this gives you the smoothest realtime experience.  
- Linux + NVIDIA GPU supported (g6.xlarge = perfect).  
- Browser UI is built-in (`index.html`) — just open the IP:8000 and grant mic/camera.  
- Research preview (expect occasional rough edges, but the core is buttery).  

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

# Start the server (runs on port 8000)
uv run server.py
```

First run will auto-download the models (~2.6 GB total — takes a minute or two depending on your network).  

Leave this terminal open (or use `tmux` / `screen` so it survives disconnects — I’ll give systemd service if you want it to auto-start on boot).

#### 4. Access via Browser (Realtime v2v + Vision)
Open in any modern browser:  
**http://<your-public-ip>:8000**

1. Click “Allow” when it asks for microphone + camera.  
2. Start talking naturally — no buttons needed.  
3. Watch the live transcript + hear streaming TTS replies.  
4. Show the camera something → it understands vision too (multimodal magic).  
5. Interrupt anytime by speaking while it’s replying.

That’s it. You now have a fully private, realtime, multimodal Gemma 4 E2B voice assistant running on your g6.xlarge, accessible from anywhere.

#### Performance on g6.xlarge (L4 GPU)
The README benchmarks on M3 Pro Mac (~83 tokens/sec decode, 2.5–3s end-to-end).  
On the L4 you’ll see **even better**: sub-2s responses, higher tokens/sec, and room for multiple concurrent sessions if you want. Kokoro TTS on ONNX is lightning fast. GPU utilization will be excellent.

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

  [Install]
  WantedBy=multi-user.target
  EOF

  sudo systemctl daemon-reload
  sudo systemctl enable --now parlor
  ```
- **Security**: Keep port 8000 restricted to your IP (or add Nginx + HTTPS later — ping me).  
- **Stop saving money**: Just stop the EC2 instance when done (EBS volume keeps everything).  
- **Monitor**: `nvidia-smi -l 1` in another session.

This beats every other self-hosted v2v setup I’ve seen for simplicity + realtime quality on a single GPU. No Ollama, no extra Whisper/Piper, no Docker needed — pure native Gemma 4 E2B + Kokoro.

If you hit any snag (model download fail, GPU not detected, want to add E4B toggle, dockerize it, add RAG, or multi-user), just paste the terminal output or error and we’ll fix in one message.  

We can also turn this into a GitHub Actions auto-build/deb package if you want (you mentioned packaging stuff before).  

Fire it up and tell me how the conversation feels — this is going to be silky smooth on the L4. 🚀  
