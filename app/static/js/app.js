/**
 * Amharic Speech AI Studio — Frontend JavaScript
 */

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  checkSystemStatus();
  initSTT();
  initTTS();
  initStudioRecorder();
});

/* ─── 1. Navigation Tabs ─────────────────────────────────────────────────── */
function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabKey = btn.getAttribute('data-tab');

      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetContent = document.getElementById(`tab-${tabKey}`);
      if (targetContent) targetContent.classList.add('active');
    });
  });
}

/* ─── 2. System Status ───────────────────────────────────────────────────── */
async function checkSystemStatus() {
  const statusEl = document.getElementById('system-status-text');
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    statusEl.textContent = `${data.mode}`;
  } catch (err) {
    statusEl.textContent = 'Server Offline';
  }
}

/* ─── 3. Speech-to-Text (STT) ────────────────────────────────────────────── */
function initSTT() {
  const recordBtn = document.getElementById('mic-record-btn');
  const recordLabel = document.getElementById('record-label');
  const fileInput = document.getElementById('audio-upload-input');
  const fileNameDisplay = document.getElementById('selected-file-name');
  const outputArea = document.getElementById('stt-output');
  const latencyEl = document.getElementById('stt-latency');
  const copyBtn = document.getElementById('copy-transcript-btn');

  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;

  // Microphone recording
  recordBtn.addEventListener('click', async () => {
    if (!isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
          await sendAudioForSTT(audioBlob);
        };

        mediaRecorder.start();
        isRecording = true;
        recordBtn.classList.add('recording');
        recordLabel.textContent = 'ቀረጻውን አቁም (Stop & Transcribe)';
        outputArea.value = 'ድምጽ እየተቀዳ ነው...';
      } catch (err) {
        alert('ማይክራፎን መክፈት አልተቻለም: ' + err.message);
      }
    } else {
      mediaRecorder.stop();
      isRecording = false;
      recordBtn.classList.remove('recording');
      recordLabel.textContent = 'ድምጽ መቅረጽ ጀምር';
      outputArea.value = 'ድምጹን ወደ ጽሑፍ እየተቀየረ ነው...';
    }
  });

  // File upload
  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
      fileNameDisplay.textContent = file.name;
      outputArea.value = 'ፋይሉ እየተተረጎመ ነው...';
      await sendAudioForSTT(file);
    }
  });

  async function sendAudioForSTT(blobOrFile) {
    const formData = new FormData();
    formData.append('audio', blobOrFile);

    try {
      const res = await fetch('/api/stt', { method: 'POST', body: formData });
      const data = await res.json();

      if (data.success) {
        outputArea.value = data.transcript || '(ድምፅ አልተለየም)';
        latencyEl.textContent = `የፈጀው ጊዜ: ${data.latency_sec}s`;
      } else {
        outputArea.value = `ስህተት: ${data.error}`;
      }
    } catch (err) {
      outputArea.value = `የግንኙነት ስህተት: ${err.message}`;
    }
  }

  // Copy button
  copyBtn.addEventListener('click', () => {
    if (outputArea.value) {
      navigator.clipboard.writeText(outputArea.value);
      copyBtn.textContent = 'ተቀድቷል! ✓';
      setTimeout(() => copyBtn.textContent = 'ቅዳ (Copy)', 2000);
    }
  });
}

/* ─── 4. Text-to-Speech (TTS) ────────────────────────────────────────────── */
function initTTS() {
  const inputArea = document.getElementById('tts-input');
  const genBtn = document.getElementById('tts-generate-btn');
  const playerWrapper = document.getElementById('tts-player-wrapper');
  const audioPlayer = document.getElementById('tts-audio-player');
  const latencyEl = document.getElementById('tts-latency');
  const chips = document.querySelectorAll('.phrase-chip');

  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      inputArea.value = chip.getAttribute('data-text');
    });
  });

  genBtn.addEventListener('click', async () => {
    const text = inputArea.value.trim();
    if (!text) {
      alert('እባክዎ ጽሑፍ ያስገቡ!');
      return;
    }

    genBtn.disabled = true;
    genBtn.textContent = 'ድምፅ እየተፈጠረ ነው...';

    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await res.json();

      if (data.success) {
        audioPlayer.src = data.audio_url + '?t=' + Date.now();
        playerWrapper.style.display = 'block';
        latencyEl.textContent = `የፈጀው ጊዜ: ${data.latency_sec}s`;
        audioPlayer.play();
      } else {
        alert('ስህተት: ' + data.error);
      }
    } catch (err) {
      alert('የግንኙነት ስህተት: ' + err.message);
    } finally {
      genBtn.disabled = false;
      genBtn.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg> ድምፅ ፍጠር (Synthesize Audio)`;
    }
  });
}

/* ─── 5. Recording Studio ────────────────────────────────────────────────── */
function initStudioRecorder() {
  const recordBtn = document.getElementById('studio-record-btn');
  const label = document.getElementById('studio-record-label');
  const promptInput = document.getElementById('record-text-prompt');
  const previewBox = document.getElementById('studio-audio-preview');
  const player = document.getElementById('studio-audio-player');
  const saveBtn = document.getElementById('studio-save-btn');
  const statusMsg = document.getElementById('studio-save-status');

  let mediaRecorder = null;
  let chunks = [];
  let isRecording = false;
  let currentBlob = null;

  recordBtn.addEventListener('click', async () => {
    if (!isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        chunks = [];

        mediaRecorder.ondataavailable = e => chunks.push(e.data);
        mediaRecorder.onstop = () => {
          currentBlob = new Blob(chunks, { type: 'audio/wav' });
          player.src = URL.createObjectURL(currentBlob);
          previewBox.style.display = 'block';
        };

        mediaRecorder.start();
        isRecording = true;
        recordBtn.classList.add('recording');
        label.textContent = '⏹️ ቀረጻውን አቁም (Stop)';
      } catch (err) {
        alert('ስህተት: ' + err.message);
      }
    } else {
      mediaRecorder.stop();
      isRecording = false;
      recordBtn.classList.remove('recording');
      label.textContent = '🎙️ ቀረጻ ጀምር (Record)';
    }
  });

  saveBtn.addEventListener('click', async () => {
    if (!currentBlob || !promptInput.value.trim()) return;

    const fd = new FormData();
    fd.append('audio', currentBlob);
    fd.append('text', promptInput.value.trim());

    saveBtn.disabled = true;
    saveBtn.textContent = 'እየተመዘገበ ነው...';

    try {
      const res = await fetch('/api/record', { method: 'POST', body: fd });
      const data = await res.json();
      if (data.success) {
        statusMsg.textContent = '✅ ' + data.message;
        statusMsg.style.color = '#10b981';
        previewBox.style.display = 'none';
        currentBlob = null;
      }
    } catch (err) {
      statusMsg.textContent = '❌ ስህተት: ' + err.message;
      statusMsg.style.color = '#ef4444';
    } finally {
      saveBtn.disabled = false;
      saveBtn.textContent = '💾 ዳታውን መዝግብ (Save to Dataset)';
    }
  });
}
