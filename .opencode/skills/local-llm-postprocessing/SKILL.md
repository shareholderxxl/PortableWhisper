# Local LLM Post-Processing — Phase 3

## Ziel
Nach der Whisper-Transkription wird der Rohtext durch ein lokales LLM (Qwen 3 2B / Gemma 4 2B) bereinigt: Stotterer entfernen, Grammatik korrigieren, Interpunktion setzen. Das LLM läuft **autark im Python-Backend** (kein Ollama, keine Cloud-API).

---

## 1. Modellauswahl

| Modell | Parameter | ONNX? | Speicher | Qualität |
|--------|-----------|-------|----------|----------|
| **Qwen 3 2B** | 2,2 Mrd | Ja (via Optimum) | ~4 GB RAM | ⭐⭐⭐ |
| **Gemma 4 2B** | 2,6 Mrd | Ja (Keras) | ~5 GB RAM | ⭐⭐⭐ |
| **Phi-3-mini** | 3,8 Mrd | Ja (ONNX) | ~8 GB RAM | ⭐⭐⭐⭐ |

**Empfehlung:** Qwen 3 2B — klein, ONNX-kompatibel, für Textkorrektur ausreichend.

---

## 2. ONNX-Export oder fertiges ONNX-Modell

### Variante A: HuggingFace ONNX-Modell laden

```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-2B"  # Prüfen, ob ONNX-Version existiert

# ONNX Runtime Session
model = ORTModelForCausalLM.from_pretrained(
    MODEL_ID,
    export=True,              # Automatischer ONNX-Export beim ersten Laden
    provider="CPUExecutionProvider",  # Oder DmlExecutionProvider (Phase 2)
    use_cache=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### Variante B: Direkter ONNX-Export mit optimum-cli

```bash
# Einmaliger Export (auf Build-Maschine)
optimum-cli export onnx --model Qwen/Qwen3-2B qwen3-2b-onnx/
```

Dann im Backend:

```python
model = ORTModelForCausalLM.from_pretrained(
    "models/qwen3-2b-onnx",
    provider="DmlExecutionProvider",
)
```

---

## 3. Prompt-Templates (aus dem UI konfigurierbar)

### Benannte Prompts (vom Nutzer erstellbar)

```python
PROMPT_TEMPLATES = {
    "standard-korrektur": (
        "Du bist ein Transkriptions-Korrektur-Assistent. "
        "Korrigiere den folgenden Rohtext: Entferne Füllwörter (ähm, also, sozusagen), "
        "setze korrekte Interpunktion, korrigiere offensichtliche Grammatikfehler. "
        "Behalte den Inhalt und Stil bei. Ändere keine Fachbegriffe.\n\n"
        "Rohtext: {raw_text}\n\n"
        "Korrigierter Text:"
    ),
    "protokoll-stil": (
        "Du bist ein Protokoll-Assistent. Formuliere den folgenden Rohtext "
        "in einen klaren, prägnanten Protokollstil um. Verwende nominalisierte Formen, "
        "entferne Redundanzen, behalte alle Fakten.\n\n"
        "Rohtext: {raw_text}\n\n"
        "Protokoll:"
    ),
    "minimal": (
        "Korrigiere nur offensichtliche Transkriptionsfehler im folgenden Text. "
        "Ändere nichts am Inhalt oder Stil. Setze lediglich Punkte und Kommas.\n\n"
        "{raw_text}\n\n"
        "Korrigiert:"
    ),
}
```

---

## 4. Inferenz-Engine

### `backend/llm_engine.py`

```python
import re
import time
from typing import Optional
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

class PostProcessingEngine:
    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        device_id: str = "0",
    ):
        self.model = ORTModelForCausalLM.from_pretrained(
            model_path,
            provider=provider,
            use_cache=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.current_prompt = "standard-korrektur"

    def set_prompt_template(self, template_name: str, custom_text: Optional[str] = None):
        """Setzt das aktive Prompt-Template (aus den UI-Einstellungen)."""
        if custom_text:
            self.current_prompt = custom_text
        else:
            self.current_prompt = template_name

    def postprocess(self, raw_text: str, max_new_tokens: int = 256) -> dict:
        """Wendet das LLM auf den Rohtext an."""
        start = time.time()

        prompt = self.current_prompt.format(raw_text=raw_text)

        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,       # Niedrige Temperatur für deterministische Korrektur
            top_p=0.9,
            do_sample=True,        # Leichte Varianz erlaubt
            repetition_penalty=1.1,
        )

        corrected = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        elapsed = time.time() - start

        return {
            "original": raw_text,
            "corrected": corrected,
            "prompt_used": self.current_prompt,
            "inference_ms": round(elapsed * 1000),
            "tokens_generated": len(outputs[0]) - inputs.input_ids.shape[1],
        }
```

---

## 5. FastAPI-Endpunkt

### `backend/main.py` (Erweiterung)

```python
from pydantic import BaseModel
from llm_engine import PostProcessingEngine

llm_engine: Optional[PostProcessingEngine] = None

class CorrectionRequest(BaseModel):
    text: str
    prompt_name: str = "standard-korrektur"
    custom_prompt: Optional[str] = None
    max_tokens: int = 256

class CorrectionResponse(BaseModel):
    original: str
    corrected: str
    inference_ms: int

@app.on_event("startup")
async def startup():
    global llm_engine
    llm_engine = PostProcessingEngine(
        model_path="models/qwen3-2b-onnx",
        provider="DmlExecutionProvider",  # NPU wenn verfügbar
    )

@app.get("/llm/prompts")
async def list_prompts():
    """Listet verfügbare Prompt-Templates."""
    return {
        "builtin": list(PROMPT_TEMPLATES.keys()),
        "custom": []  # Aus Nutzer-Konfig laden
    }

@app.post("/llm/correct", response_model=CorrectionResponse)
async def correct_text(req: CorrectionRequest):
    """Korrigiert Rohtext mit dem LLM."""
    llm_engine.set_prompt_template(req.prompt_name, req.custom_prompt)
    result = llm_engine.postprocess(req.text, max_new_tokens=req.max_tokens)
    return CorrectionResponse(
        original=result["original"],
        corrected=result["corrected"],
        inference_ms=result["inference_ms"],
    )
```

---

## 6. Frontend: Post-Processing Tab

### `src/components/PostProcessingTab.tsx`

```tsx
import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';

interface PromptTemplate {
  name: string;
  systemPrompt: string;
}

export const PostProcessingTab: React.FC = () => {
  const [prompts, setPrompts] = useState<PromptTemplate[]>([]);
  const [selectedPrompt, setSelectedPrompt] = useState('standard-korrektur');
  const [customPrompt, setCustomPrompt] = useState('');

  useEffect(() => {
    // Prompts vom Backend laden
    fetch('http://127.0.0.1:8765/llm/prompts')
      .then(res => res.json())
      .then(data => {
        const builtin = data.builtin.map((name: string) => ({
          name,
          systemPrompt: getDefaultPrompt(name),
        }));
        setPrompts([...builtin, ...data.custom]);
      });
  }, []);

  const handleSavePrompt = () => {
    // Custom Prompt speichern (lokal oder ans Backend senden)
    setPrompts(prev => [
      ...prev,
      { name: `Custom #${prev.length + 1}`, systemPrompt: customPrompt },
    ]);
    setCustomPrompt('');
  };

  return (
    <div style={{ padding: 16 }}>
      <h2>Post-Processing</h2>
      <p>Wähle ein Prompt-Template für die LLM-Korrektur:</p>

      <select
        value={selectedPrompt}
        onChange={e => setSelectedPrompt(e.target.value)}
        style={{ width: '100%', marginBottom: 12, padding: 8 }}
      >
        {prompts.map(p => (
          <option key={p.name} value={p.name}>{p.name}</option>
        ))}
      </select>

      <h3>System-Prompt:</h3>
      <textarea
        value={
          prompts.find(p => p.name === selectedPrompt)?.systemPrompt || ''
        }
        readOnly
        rows={8}
        style={{ width: '100%', fontFamily: 'monospace', fontSize: 12 }}
      />

      <h3>Neuen Prompt erstellen:</h3>
      <textarea
        value={customPrompt}
        onChange={e => setCustomPrompt(e.target.value)}
        placeholder="System-Prompt Text hier eingeben... Verwende {raw_text} als Platzhalter für den Rohtext."
        rows={6}
        style={{ width: '100%', fontFamily: 'monospace', fontSize: 12 }}
      />
      <button onClick={handleSavePrompt} style={{ marginTop: 8, padding: '8px 16px' }}>
        Prompt speichern
      </button>
    </div>
  );
};

function getDefaultPrompt(name: string): string {
  const prompts: Record<string, string> = {
    'standard-korrektur': 'Du bist ein Transkriptions-Korrektur-Assistent...',
    'protokoll-stil': 'Du bist ein Protokoll-Assistent...',
    'minimal': 'Korrigiere nur offensichtliche Transkriptionsfehler...',
  };
  return prompts[name] || '';
}
```

---

## 7. Vollständiger Workflow

```text
Mikrofon → WebRTC (Frontend) → HTTP/WebSocket → Whisper-ONNX (Backend) → Rohtext
                                                                              ↓
                                                                     LLM-Engine (ONNX)
                                                                         ↓
                                                                    Korrigierter Text
                                                                         ↓
                                                               Clipboard-Injection
```

---

## 8. Wichtige Fallstricke

| Problem | Lösung |
|---------|--------|
| **LLM zu groß für ONNX/Portable** | Ab Phase 3: Nur 2B-Modelle, kein 7B+. Export mit `optimum-cli` optimieren |
| **Inferenz zu langsam** | NPU-Beschleunigung (Phase 2) + KV-Cache + Batch-Größe 1 |
| **ONNX-Export fehlschlägt** | Prüfen ob `optimum>=1.23` installiert; manueller Export per optimum-cli |
| **Prompt-Injection** | Kein nutzereingegebener Prompt enthält Code-Ausführung — reiner Text |
| **Clipboard nicht erreichbar** | Tauri `clipboard-manager` Plugin verwenden |
| **Modell-Download > 2 GB** | Nicht in PyInstaller einbetten — Modell beim ersten Start laden |