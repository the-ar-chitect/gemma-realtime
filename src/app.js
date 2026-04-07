const $ = id => document.getElementById(id);
const video = $('video'), cameraToggle = $('cameraToggle');
const micToggle = $('micToggle'), connectToggle = $('connectToggle');
const messagesDiv = $('messages'), statusEl = $('status');
const stateDot = $('stateDot'), stateText = $('stateText');
const viewportWrap = $('viewportWrap');
const waveformCanvas = $('waveform');
const waveformCtx = waveformCanvas.getContext('2d');
const maxTurnsInput = $('maxTurns');
const contextInfo = $('contextInfo');
const toolLog = $('toolLog');
const turnCounter = $('turnCounter');

let ws, mediaStream, myvad;
let cameraEnabled = true;
let micEnabled = true;
let connected = false;       // Whether we want to be connected
let audioCtx, currentSource;
let state = 'idle';
let ignoreIncomingAudio = false;

// Streaming audio playback state
let streamSampleRate = 24000;
let streamNextTime = 0;         // When to schedule next chunk
let streamSources = [];         // Active AudioBufferSourceNodes
let streamTtsTime = null;

// ── Waveform Visualizer ──
let analyser, micSource;
const BAR_COUNT = 40;
const BAR_GAP = 3;
let waveformRAF;
let ambientPhase = 0;

function initWaveformCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = waveformCanvas.getBoundingClientRect();
  waveformCanvas.width = rect.width * dpr;
  waveformCanvas.height = rect.height * dpr;
  waveformCtx.scale(dpr, dpr);
}

function getStateColor() {
  const colors = { listening: '#4ade80', processing: '#f59e0b', speaking: '#818cf8', loading: '#3a3d46', idle: '#3a3d46' };
  return colors[state] || colors.idle;
}

function drawWaveform() {
  const w = waveformCanvas.getBoundingClientRect().width;
  const h = waveformCanvas.getBoundingClientRect().height;
  waveformCtx.clearRect(0, 0, w, h);

  const barWidth = (w - (BAR_COUNT - 1) * BAR_GAP) / BAR_COUNT;
  const color = getStateColor();
  waveformCtx.fillStyle = color;

  let dataArray = null;
  if (analyser) {
    dataArray = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(dataArray);
  }

  for (let i = 0; i < BAR_COUNT; i++) {
    let amplitude;
    if (dataArray) {
      // Map bar index to frequency bin
      const binIndex = Math.floor((i / BAR_COUNT) * dataArray.length * 0.6);
      amplitude = dataArray[binIndex] / 255;
    }

    // If no real audio data or very quiet, use ambient drift
    if (!dataArray || amplitude < 0.02) {
      ambientPhase += 0.0001;
      const drift = Math.sin(ambientPhase * 3 + i * 0.4) * 0.5 + 0.5;
      amplitude = 0.03 + drift * 0.04;
    }

    const barH = Math.max(2, amplitude * (h - 4));
    const x = i * (barWidth + BAR_GAP);
    const y = (h - barH) / 2;

    waveformCtx.globalAlpha = 0.3 + amplitude * 0.7;
    waveformCtx.beginPath();
    const r = Math.min(barWidth / 2, barH / 2, 3);
    waveformCtx.roundRect(x, y, barWidth, barH, r);
    waveformCtx.fill();
  }

  waveformCtx.globalAlpha = 1;
  waveformRAF = requestAnimationFrame(drawWaveform);
}

// ── Dynamic glow intensity for speaking state ──
function updateSpeakingGlow() {
  if (state !== 'speaking' || !analyser) return;
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(data);
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i];
  const avg = sum / data.length / 255;
  // Modulate the glow layers based on audio amplitude
  const intensity = 0.3 + avg * 0.7;
  viewportWrap.style.setProperty('--speak-intensity', intensity);
  const spread = 20 + avg * 60;
  const inner = 15 + avg * 25;
  viewportWrap.querySelector('.viewport-glow').style.boxShadow =
    `0 0 ${spread}px ${spread * 0.4}px rgba(129,140,248,${intensity * 0.25})`;
  viewportWrap.style.boxShadow =
    `inset 0 0 ${inner}px rgba(129,140,248,${intensity * 0.15}), 0 0 ${inner}px rgba(129,140,248,${intensity * 0.2})`;
  requestAnimationFrame(updateSpeakingGlow);
}

// ── State Machine ──
function setState(newState) {
  state = newState;

  // Update state indicator
  stateDot.className = `dot ${newState}`;
  const labels = { idle: 'Idle', loading: 'Loading...', listening: 'Listening', processing: 'Thinking...', speaking: 'Speaking' };
  stateText.textContent = labels[newState] || newState;

  // Update viewport glow class
  viewportWrap.className = `viewport-wrap ${newState === 'idle' ? 'loading' : newState}`;

  // Reset inline styles from speaking glow
  if (newState !== 'speaking') {
    viewportWrap.style.boxShadow = '';
    viewportWrap.querySelector('.viewport-glow').style.boxShadow = '';
  }

  // Update CSS custom properties for state color (must use resolved values, not nested var())
  const stateVars = {
    listening: ['#4ade80', 'rgba(74,222,128,0.12)'],
    processing: ['#f59e0b', 'rgba(245,158,11,0.12)'],
    speaking: ['#818cf8', 'rgba(129,140,248,0.12)'],
    loading: ['#3a3d46', 'rgba(58,61,70,0.12)'],
    idle: ['#3a3d46', 'rgba(58,61,70,0.12)'],
  };
  const [glow, glowDim] = stateVars[newState] || stateVars.idle;
  document.documentElement.style.setProperty('--glow', glow);
  document.documentElement.style.setProperty('--glow-dim', glowDim);

  // Kick off speaking glow animation
  if (newState === 'speaking') requestAnimationFrame(updateSpeakingGlow);

  // Raise VAD threshold during speaking to reduce echo false triggers
  if (myvad) {
    myvad.setOptions({ positiveSpeechThreshold: newState === 'speaking' ? 0.92 : 0.5 });
  }

  // Connect/disconnect mic analyser based on state
  if (newState === 'listening' && mediaStream && audioCtx && analyser) {
    if (!micSource) {
      micSource = audioCtx.createMediaStreamSource(mediaStream);
    }
    try { micSource.connect(analyser); } catch {}
  } else if (micSource && newState !== 'listening') {
    try { micSource.disconnect(analyser); } catch {}
  }
}

// ── WebSocket ──
let reconnectTimer = null;

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  connected = true;
  connectToggle.textContent = 'Disconnect';
  connectToggle.classList.add('danger');
  connectToggle.classList.remove('active');
  setState('loading');

  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => {
    setStatus('connected', 'Connected');
    setState('listening');
    if (myvad && micEnabled) myvad.start();
  };
  ws.onclose = () => {
    setStatus('disconnected', 'Disconnected');
    if (connected) {
      // Unexpected close — reconnect
      reconnectTimer = setTimeout(connect, 2000);
    } else {
      setState('idle');
    }
  };
  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    if (msg.type === 'tool_call') {
      showToolCall(msg.tool, msg.reason);
    } else if (msg.type === 'text') {
      if (msg.transcription) {
        const userMsgs = messagesDiv.querySelectorAll('.msg.user');
        const lastUserMsg = userMsgs[userMsgs.length - 1];
        if (lastUserMsg) {
          const meta = lastUserMsg.querySelector('.meta');
          lastUserMsg.innerHTML = `${msg.transcription}${meta ? meta.outerHTML : ''}`;
        }
      }
      let metaText = `LLM ${msg.llm_time}s`;
      if (msg.tools_called && msg.tools_called.length) {
        metaText += ` · Tools: ${msg.tools_called.join(', ')}`;
      }
      addMessage('assistant', msg.text, metaText);
      if (msg.turn != null) {
        turnCounter.textContent = `Turn ${msg.turn}/${msg.max_turns}`;
      }
    } else if (msg.type === 'audio_start') {
      if (ignoreIncomingAudio) return;
      streamSampleRate = msg.sample_rate || 24000;
      startStreamPlayback();
    } else if (msg.type === 'audio_chunk') {
      if (ignoreIncomingAudio) return;
      queueAudioChunk(msg.audio);
    } else if (msg.type === 'audio_end') {
      if (ignoreIncomingAudio) {
        ignoreIncomingAudio = false;
        stopPlayback();
        setState('listening');
        return;
      }
      streamTtsTime = msg.tts_time;
      const meta = messagesDiv.querySelector('.msg.assistant:last-child .meta');
      if (meta) meta.textContent += ` · TTS ${msg.tts_time}s`;
    }
  };
}

function disconnect() {
  connected = false;
  connectToggle.textContent = 'Connect';
  connectToggle.classList.remove('danger');
  if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  if (myvad) myvad.pause();
  stopPlayback();
  if (ws) {
    ws.onclose = null; // Prevent reconnect
    ws.close();
    ws = null;
  }
  setStatus('disconnected', 'Disconnected');
  setState('idle');
}

// ── Tool Call Display ──
function showToolCall(toolNum, reason) {
  toolLog.hidden = false;
  const tag = document.createElement('div');
  tag.className = 'tool-tag';
  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  tag.innerHTML = `Tool ${toolNum} <span class="tool-time">${now}</span>`;
  tag.title = reason;
  toolLog.appendChild(tag);
  // Keep max 20 tags
  while (toolLog.children.length > 20) toolLog.removeChild(toolLog.firstChild);
}

function setStatus(cls, text) { statusEl.className = `status-pill ${cls}`; statusEl.textContent = text; }

// ── Camera ──
async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    video.srcObject = mediaStream;
    return;
  } catch (e) { console.warn('Video+audio failed:', e.message); }

  const streams = await Promise.allSettled([
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } }),
    navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } }),
  ]);
  mediaStream = new MediaStream();
  streams.forEach(r => { if (r.status === 'fulfilled') r.value.getTracks().forEach(t => mediaStream.addTrack(t)); });
  if (mediaStream.getVideoTracks().length) video.srcObject = mediaStream;
  if (!mediaStream.getAudioTracks().length) { cameraEnabled = false; }
}

function captureFrame() {
  if (!cameraEnabled || !video.videoWidth) return null;
  const canvas = document.createElement('canvas');
  const scale = 320 / video.videoWidth;
  canvas.width = 320; canvas.height = video.videoHeight * scale;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
}

// ── VAD Handlers ──
let speakingStartedAt = 0;
const BARGE_IN_GRACE_MS = 800; // Ignore VAD triggers shortly after TTS starts (echo)

function handleSpeechStart() {
  if (state === 'speaking') {
    // Ignore echo: don't allow barge-in right after TTS starts playing
    if (Date.now() - speakingStartedAt < BARGE_IN_GRACE_MS) {
      console.log('Barge-in suppressed (echo grace period)');
      return;
    }
    stopPlayback();
    ignoreIncomingAudio = true;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'interrupt' }));
    }
    setState('listening');
    console.log('Barge-in: interrupted playback');
  }
}

function handleSpeechEnd(audio) {
  if (state !== 'listening' || !micEnabled) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  const wavBase64 = float32ToWavBase64(audio);
  const imageBase64 = captureFrame();

  setState('processing');
  setStatus('processing', 'Processing');
  addMessage('user', '<span class="loading-dots"><span></span><span></span><span></span></span>', imageBase64 ? 'with camera' : '');

  const payload = { audio: wavBase64 };
  if (imageBase64) payload.image = imageBase64;
  ws.send(JSON.stringify(payload));
}

// ── Float32 @ 16kHz → WAV base64 ──
function float32ToWavBase64(samples) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0,'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); w(8,'WAVE'); w(12,'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 16000, true); v.setUint32(28, 32000, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); w(36,'data'); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  const bytes = new Uint8Array(buf);
  let bin = ''; for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

// ── Streaming Playback ──
function stopPlayback() {
  for (const src of streamSources) {
    try { src.stop(); } catch {}
  }
  streamSources = [];
  currentSource = null;
  streamNextTime = 0;
}

function ensureAudioCtx() {
  if (!audioCtx) {
    audioCtx = new AudioContext();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.75;
  }
}

function startStreamPlayback() {
  stopPlayback();
  ensureAudioCtx();
  if (audioCtx.state === 'suspended') audioCtx.resume();
  streamNextTime = audioCtx.currentTime + 0.05; // Small initial buffer
  speakingStartedAt = Date.now();
  setState('speaking');
}

function queueAudioChunk(base64Pcm) {
  ensureAudioCtx();

  // Decode base64 -> Int16 PCM -> Float32
  const bin = atob(base64Pcm);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const int16 = new Int16Array(bytes.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

  // Create AudioBuffer and schedule it
  const audioBuffer = audioCtx.createBuffer(1, float32.length, streamSampleRate);
  audioBuffer.getChannelData(0).set(float32);

  const source = audioCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioCtx.destination);
  source.connect(analyser);

  // Schedule gapless playback
  const startAt = Math.max(streamNextTime, audioCtx.currentTime);
  source.start(startAt);
  streamNextTime = startAt + audioBuffer.duration;

  streamSources.push(source);
  currentSource = source;

  // Clean up when this chunk finishes
  source.onended = () => {
    const idx = streamSources.indexOf(source);
    if (idx !== -1) streamSources.splice(idx, 1);
    // If this was the last chunk and no more are queued, return to listening
    if (streamSources.length === 0 && state === 'speaking') {
      currentSource = null;
      setState('listening');
      setStatus('connected', 'Connected');
    }
  };
}

// ── UI ──
function addMessage(role, text, meta) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `${text}${meta ? `<div class="meta">${meta}</div>` : ''}`;
  messagesDiv.appendChild(div);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

cameraToggle.addEventListener('click', () => {
  cameraEnabled = !cameraEnabled;
  cameraToggle.classList.toggle('active', cameraEnabled);
  cameraToggle.textContent = cameraEnabled ? 'Camera On' : 'Camera Off';
  video.style.opacity = cameraEnabled ? 1 : 0.3;
});

micToggle.addEventListener('click', () => {
  micEnabled = !micEnabled;
  micToggle.classList.toggle('active', micEnabled);
  micToggle.textContent = micEnabled ? 'Mic On' : 'Mic Off';
  if (myvad) {
    if (micEnabled && connected) myvad.start();
    else myvad.pause();
  }
});

connectToggle.addEventListener('click', () => {
  if (connected) disconnect();
  else connect();
});

// ── Init ──
async function init() {
  initWaveformCanvas();
  window.addEventListener('resize', initWaveformCanvas);

  await startCamera();

  // Fetch server config
  try {
    const cfg = await fetch('/config').then(r => r.json());
    maxTurnsInput.value = cfg.max_history_turns;
    contextInfo.textContent = `${Math.round(cfg.context_window / 1024)}K ctx`;
  } catch {}

  // Initialize VAD with shared mic stream (but don't start until connected)
  myvad = await vad.MicVAD.new({
    getStream: async () => new MediaStream(mediaStream.getAudioTracks()),
    positiveSpeechThreshold: 0.5,
    negativeSpeechThreshold: 0.25,
    redemptionMs: 600,
    minSpeechMs: 300,
    preSpeechPadMs: 300,
    onSpeechStart: handleSpeechStart,
    onSpeechEnd: handleSpeechEnd,
    onVADMisfire: () => { console.log('VAD misfire (too short)'); },
    onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
    baseAssetPath: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/",
  });

  // Don't auto-connect — wait for user to click Connect
  // myvad stays paused until connect()

  // Init audio context on first user gesture for mic visualizer
  const initAudio = () => {
    ensureAudioCtx();
    if (audioCtx.state === 'suspended') audioCtx.resume();
    document.removeEventListener('click', initAudio);
    document.removeEventListener('keydown', initAudio);
  };
  document.addEventListener('click', initAudio);
  document.addEventListener('keydown', initAudio);
  ensureAudioCtx();

  setState('idle');

  // Start waveform loop
  drawWaveform();

  console.log('Parlor ready — click Connect to start');
}

init();
