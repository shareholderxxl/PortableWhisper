# Tauri Audio Recording — Mikrofonzugriff & Audio-Streaming

## Ziel
Die Tauri-App nimmt Mikrofon-Audio auf und sendet die Audio-Chunks an das Python-FastAPI-Backend (Sidecar) zur Transkription. Nutzt die **Web Audio API** im Frontend und sendet per HTTP an `localhost:8765`.

---

## 1. Frontend: Mikrofonzugriff (WebRTC)

### `src/hooks/useAudioRecorder.ts`

```typescript
import { useState, useRef, useCallback } from 'react';

interface AudioRecorderState {
  isRecording: boolean;
  isPaused: boolean;
  durationMs: number;
  audioLevel: number;  // 0.0 - 1.0 für Visualisierung
}

export function useAudioRecorder(backendUrl: string = 'http://127.0.0.1:8765') {
  const [state, setState] = useState<AudioRecorderState>({
    isRecording: false,
    isPaused: false,
    durationMs: 0,
    audioLevel: 0,
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const chunkIntervalRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);

  const startRecording = useCallback(async () => {
    try {
      // Mikrofon-Zugriff anfordern
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      // AudioContext für Pegel-Messung
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      // MediaRecorder: Audio in Chunks aufnehmen
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });

      mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0) {
          await sendAudioChunk(event.data, backendUrl);
        }
      };

      mediaRecorder.start(1000);  // Chunk alle 1 Sekunde
      mediaRecorderRef.current = mediaRecorder;
      startTimeRef.current = Date.now();

      setState(prev => ({
        ...prev,
        isRecording: true,
        isPaused: false,
        durationMs: 0,
      }));

      // Pegel-Messung starten
      startLevelMetering();

    } catch (err) {
      console.error('Mikrofonzugriff verweigert:', err);
      throw err;
    }
  }, [backendUrl]);

  const stopRecording = useCallback(() => {
    // Letzten Chunk erzwingen
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }

    // Aufräumen
    streamRef.current?.getTracks().forEach(track => track.stop());
    audioContextRef.current?.close();
    if (chunkIntervalRef.current) {
      clearInterval(chunkIntervalRef.current);
    }

    setState(prev => ({
      ...prev,
      isRecording: false,
      isPaused: false,
      durationMs: Date.now() - startTimeRef.current,
      audioLevel: 0,
    }));
  }, []);

  const startLevelMetering = () => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const updateLevel = () => {
      if (!analyserRef.current) return;
      analyserRef.current.getByteTimeDomainData(dataArray);

      let max = 0;
      for (let i = 0; i < bufferLength; i++) {
        const value = Math.abs(dataArray[i] - 128) / 128;
        if (value > max) max = value;
      }

      setState(prev => ({ ...prev, audioLevel: max }));
      requestAnimationFrame(updateLevel);
    };
    updateLevel();
  };

  const sendAudioChunk = async (blob: Blob, url: string) => {
    try {
      const formData = new FormData();
      formData.append('file', blob, `chunk-${Date.now()}.webm`);
      formData.append('format', 'webm');

      await fetch(`${url}/transcribe`, {
        method: 'POST',
        body: formData,
      });
      // Transkriptionsergebnis wird separat über Event geholt
    } catch (err) {
      console.error('Fehler beim Senden des Audio-Chunks:', err);
    }
  };

  return {
    ...state,
    startRecording,
    stopRecording,
  };
}
```

---

## 2. Alternative: Desktop-Mikrofon via Rust/Tauri-Plugin

Für bessere Latenz und native Audio-APIs (WASAPI unter Windows):

```rust
// src-tauri/src/audio.rs (optional, falls benötigt)
// Nutzt cpal oder tauri-plugin-media für nativen Audio-Zugriff
```

Fürs Erste reicht die WebRTC-Lösung — sie funktioniert plattformunabhängig und ist einfacher.

---

## 3. Visualisierungskomponente

### `src/components/AudioVisualizer.tsx`

```tsx
import React from 'react';

interface Props {
  audioLevel: number;
  isRecording: boolean;
}

export const AudioVisualizer: React.FC<Props> = ({ audioLevel, isRecording }) => {
  const bars = 7;
  const heights = Array.from({ length: bars }, (_, i) => {
    const threshold = (i + 1) / bars;
    return audioLevel > threshold ? 100 : Math.max(10, (audioLevel / threshold) * 100);
  });

  return (
    <div style={{ display: 'flex', gap: 4, alignItems: 'center', height: 40 }}>
      {heights.map((h, i) => (
        <div
          key={i}
          style={{
            width: 8,
            height: `${isRecording ? h : 10}%`,
            backgroundColor: isRecording ? '#22c55e' : '#888',
            borderRadius: 4,
            transition: 'height 0.1s ease',
          }}
        />
      ))}
      <span style={{ marginLeft: 8, color: isRecording ? '#22c55e' : '#888', fontSize: 12 }}>
        {isRecording ? 'REC' : 'Bereit'}
      </span>
    </div>
  );
};
```

---

## 4. Backend: Audio-Chunk empfangen

Der FastAPI-Endpunkt muss WebM-Chunks verarbeiten können (librosa/ffmpeg notwendig):

```python
import subprocess
import tempfile
import os

def convert_webm_to_wav(webm_bytes: bytes) -> str:
    """Konvertiert WebM zu 16kHz WAV via ffmpeg."""
    tmp_webm = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    tmp_webm.write(webm_bytes)
    tmp_webm.close()

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()

    subprocess.run([
        "ffmpeg", "-y", "-i", tmp_webm.name,
        "-ar", "16000", "-ac", "1",
        "-sample_fmt", "s16",
        tmp_wav.name
    ], capture_output=True)

    os.unlink(tmp_webm.name)
    return tmp_wav.name
```

**Hinweis:** `ffmpeg` muss im System installiert oder als Binary mitgeliefert werden.

---

## 5. Mikrofonberechtigung

### Tauri v2 Capabilities

```json
{
  "permissions": [
    "core:default",
    "shell:allow-open"
  ]
}
```

Der Browser/WebView fragt automatisch nach Mikrofon-Zugriff über `navigator.mediaDevices.getUserMedia()` — Tauri leitet diese Anfrage an das Betriebssystem weiter.

---

## 6. Wichtige Fallstricke

| Problem | Lösung |
|---------|--------|
| **getUserMedia() fehlschlägt** | App muss über HTTPS oder localhost laufen; Berechtigung in Windows-Einstellungen aktivieren |
| **WebM-Codec nicht supported** | Fallback auf `audio/wav` mit `MediaRecorder` (nicht alle Browser unterstützen das) |
| **Chunk-Überlappungen** | Backend muss Chunks puffern und sequenziell verarbeiten oder Intervall anpassen |
| **Latenz zu hoch** | Chunk-Intervall auf 500ms reduzieren; Streaming-Modus (WebSocket) implementieren |
| **Keine Transkription nach Stop** | Letzten Chunk explizit über `requestData()` abrufen vor stop() |

---

## 7. WebSocket-Alternative (für niedrige Latenz)

Statt HTTP-Chunks: WebSocket-Verbindung für Echtzeit-Audio-Streaming.

### Backend (FastAPI + WebSocket)

```python
from fastapi import WebSocket
import json

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = b""

    while True:
        data = await websocket.receive_bytes()
        audio_buffer += data

        if len(audio_buffer) >= 32000:  # ~2 Sekunden bei 16kHz
            text = transcribe_audio_bytes(audio_buffer)
            await websocket.send_text(json.dumps({"text": text}))
            audio_buffer = b""
```

### Frontend (WebSocket-Client)

```typescript
const ws = new WebSocket('ws://127.0.0.1:8765/ws/transcribe');
ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    setTranscript(prev => prev + result.text + ' ');
};
```