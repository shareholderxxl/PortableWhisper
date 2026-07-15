# PortableWhisper — Phase 3B Plan: LLM-gestützte Korrektur (CPU-first, swappable)

**Stand:** 2026-07-15
**Branch:** phase-3b (Fortführung von phase-3 / Phase 3A)
**Status:** Implementiert (Schritt 0–5), validierung auf dem Notebook ausstehend

---

## Spike-Ergebnis & Architektur-Entscheid (Schritt 0, 2026-07-15)

Der Plan sah ursprünglich `onnxruntime-genai` + genai-Format-Modell vor. Der
Spike hat das **verworfen** — Qwen3.5-0.8B läuft stattdessen auf *plain*
onnxruntime (das ohnehin für Parakeet installiert ist):

- ✅ `onnxruntime-genai 0.14.1` würde `qwen3_5_text` unterstützen, ABER:
- ❌ Das von HF heruntergeladene `onnx-community/Qwen3.5-0.8B-ONNX` ist ein
  **optimum**-Export (Decoder-Cache-Layout: `past_conv.<i>` kernel=4,
  `past_recurrent.<i>`), **inkompatibel** mit dem genai-`qwen3_5_text`-Runtime
  (erwartet `conv_state` kernel=3, andere Namen). `og.Model()` scheitert an
  fehlendem `genai_config.json`; eine geschriebene Brücke scheitert an der
  Cache-Shape-Differenz.
- ❌ Es existiert **kein** vorgefertigtes genai-Format-Qwen3.5-0.8B auf HF;
  eigenes Bauen bräuchte torch + transformers 5.3.0.dev0.

**Entscheidung:** Plain onnxruntime + manueller Generierungs-Loop.
- Das optimum-q4-Modell lädt & läuft direkt auf `onnxruntime` (forward pass
  verifiziert: korrekte Logits `(1,S,248320)`).
- **Full-Recompute-Greedy-Decode**: jeder Token-Schritt verarbeitet die
  komplette Sequenz mit geleertem Cache neu. Korrekt, simpel (kein
  Hybrid-Cache-Management), aber O(n²). Inkrementeller KV-Cache = spätere
  Optimierung falls die CPU zu langsam.
- Tokenisierung via `tokenizers`-Bibliothek (`tokenizer.json` liegt bei).
- `/no_think` schaltet Qwen3.5-Thinking aus; `<think>…</think>` wird als
  Sicherheitsnetz entfernt.

**Offen (nur auf dem Windows-Notebook messbar):** CPU-Geschwindigkeit &
Ausgabequalität. Auf dem Dev-Server (alter Thin Client) ist ein Forward
pathologisch langsam (~120 s) — NICHT repräsentativ. Wenn die Full-Recompute-
Latenz zu hoch ist → Schritt 5b: inkrementeller KV-Cache nachrüsten.

## Implementierungsstand

| Schritt | Status | Datei |
|---|---|---|
| 0 — Feasibility Spike | ✅ | (dieses Dokument) |
| 1 — `text_correction.py` (`TextCorrector`) | ✅ | `backend/text_correction.py` |
| 2 — Integration in `main.py` (lazy, Fallback) | ✅ | `backend/main.py` (`/stop`) |
| 3 — Settings API | ✅ | `backend/main.py` (`/settings/correction`) |
| 4 — Frontend-Toggle | ✅ | `frontend/dist/index.html` |
| 5 — Dependencies & PyInstaller | ✅ | `requirements.txt`, `whisper-backend.spec` |
| 6 — NPU-Option | ⏳ später | — |

Default-Modell: `model/qwen3.5-0.8b-onnx/` (q4). Toggle default = **AN**;
Fallback liefert Rohtext, falls das Modell fehlt/schlägt fehl.

---

## Ziel
Erweiterung der Transkriptions-Pipeline um eine **LLM-Korrektur-Stufe**, die nach Parakeet (ASR) und der optionalen Phase-3A-Heuristik den Text grammatikalisch glättet, Füllwörter/Disfluenzen entfernt und Selbstkorrekturen auflöst.
**Start: Qwen3.5-0.8B auf CPU.** Modell austauschbar (→ 2B, → Qwen3-1.7B NPU). NPU-Pfad bleibt als Option erhalten.

## Bestätigte Entscheidungen
- **ASR-Engine: Parakeet TDT bleibt.** (Der "Unified Qwen3-ASR ersetzt Parakeet + Korrektur"-Ansatz wurde verworfen — Research ergab: kein zuverlässiges Editing per Prompt, NPU-Pfad schlechter als bei einem separaten Korrektur-LLM.)
- **Korrektur: Qwen3.5-0.8B ONNX, CPU via `onnxruntime-genai`** als Default.
- **Swappable per Config/Pfad:** `0.8B` ↔ `2B` (CPU) ↔ `Qwen3-1.7B` (NPU).
- **NPU** im Plan als Option, Umsetzung nach dem CPU-Spike (späterer Sub-Schritt).

## Wichtige neue Erkenntnisse (Recherche 2026-07-15)
- `onnxruntime-genai` unterstützt den Qwen3.5-Hybrid-Decoder (GatedDeltaNet + Attention) ab **PR #2043** (microsoft/onnxruntime-genai Releases). → CPU-Inferenz ist machbar, **keine "neuen ONNX-Ops" nötig** (der ältere justinchuby-Gist war nur Forschung dazu).
- ONNX-Export vorhanden: `onnx-community/Qwen3.5-0.8B-ONNX` (+ optimierte `-OPT` Variante `onnx-community/Qwen3.5-0.8B-ONNX-OPT`).
- Qualität (IFEval Instruction-Following, Non-Thinking): Qwen3-1.7B **68.2** > Qwen3.5-2B **61.2** > Qwen3.5-0.8B **52.1**. → 0.8B am unteren Limit für Instruction-Following; daher der Test-Vorbehalt (0.8B zuerst, dann 2B) sinnvoll.
- Qwen3.5-0.8B ist ein **multimodales** Modell (Vision-Encoder inkludiert). Für Text-only-Korrektur muss ein Text-only-Pfad bestätigt werden (die `-OPT`-Variante ist vermutlich text-optimiert).

## Implementierung (Schritte)

### Schritt 0 — Feasibility Spike (lokal, venv)
- `Qwen3.5-0.8B-ONNX` (oder `-OPT`) herunterladen.
- Minimal-Skript mit `onnxruntime-genai`: Korrektur-Prompt ("Korrigiere folgenden transkribierten Text auf Deutsch, entferne Füllwörter...") → Ausgabe.
- Verifizieren: (a) läuft ohne Vision-Encoder (Text-only-Pfad), (b) CPU-Geschwindigkeit bei ~100-Wort-Input, (c) Ausgabequalität auf 2–3 deutschen Diktat-Beispielen.
- Falls 0.8B zu schwach: direkt 2B testen. Ergebnis bestimmt den Default.

### Schritt 1 — Korrekturmodul (`backend/text_correction.py`, neu)
- Klasse `TextCorrector` mit `load(model_dir, provider="cpu")` und `correct(text) -> str`.
- Nutzt `onnxruntime_genai.Model` + `Tokenizer` (kein `transformers` nötig — Tokenizer ist im Modellordner).
- **Lazy-Load** beim ersten `correct()`-Aufruf (App-Start bleibt schnell).
- Prompt-Template für DE-Diktat-Korrektur (Filler entfernen, Grammatik, Selbstkorrektur), non-thinking mode.
- Fehlerabsicherung: bei Modell-Fehler Fallback auf Eingabetext (kein Crash der Pipeline).

### Schritt 2 — Integration in `backend/main.py`
- Globaler `text_corrector` (lazy) + Flag `llm_correction_enabled` (Default: True).
- Nach Transkription (und nach Phase 3A, falls aktiv): `if llm_correction_enabled: text = corrector.correct(text)`.
- Config: `correction_model_path` (Default `model/qwen3.5-0.8b-onnx/`), `correction_provider` ("cpu" | "npu").

### Schritt 3 — Settings API
- `GET/POST /settings/correction` (enabled, model_path, provider) — analog zu `GET/POST /settings/cleanup` aus Phase 3A.
- Persistenz in `config.json` (zentral, wie bisher).

### Schritt 4 — Frontend (`frontend/dist/index.html`)
- Toggle "LLM-Korrektur" (in Speech-to-Text-Sektion, unter dem 3A-Toggle).
- Modell-Dropdown: 0.8B (CPU) / 2B (CPU) / 1.7B (NPU) — Auswahl setzt `model_path`.
- JS: `toggleLlmCorrection()`, `loadLlmCorrectionSetting()` analog zu Phase 3A.

### Schritt 5 — Dependencies & PyInstaller
- `onnxruntime-genai` (Version mit PR #2043+, CPU-Build) zu requirements/pyproject.
- PyInstaller spec: `collect_all('onnxruntime_genai')` (bekannter Risiko-Punkt — Analyze-Schritt kann hängen; wie bei `onnxruntime`/`onnx-asr`).
- Modell-Dateien: Default 0.8B nach `model/qwen3.5-0.8b-onnx/` ins ZIP packen (+~0.5 GB). 2B / 1.7B-NPU als optionaler On-Demand-Download (Muster wie `gpu_manager`), sonst ZIP zu groß.

### Schritt 6 — NPU-Option (später, separat)
- `provider="npu"`: Qwen3-1.7B via VitisAI EP (AMD) / OpenVINO (Intel).
- Runtime wie GPU-Libs on-demand downloaden (`gpu_manager`-Pattern); Modell kompiliert bei erstem Load (10–30 s).
- Abstraktion in `text_correction.py`: nur EP/Provider-Wechsel, gleiche Logik.

## Risiken
1. **Multimodal-Modell:** Qwen3.5-0.8B ONNX enthält Vision-Encoder. Text-only-Pfad im Spike bestätigen (oder `-OPT` nutzen).
2. **PyInstaller + onnxruntime-genai:** noch nicht bewiesen gebündelt (siehe Research) → Spike vor Integration.
3. **0.8B Qualität:** unteres Limit (IFEval 52.1) → User-Test 0.8B → 2B.
4. **ZIP-Größe:** +0.5–2 GB → On-Demand-Download für Nicht-Default-Modelle empfohlen.
5. **onnxruntime-genai Version** muss PR #2043 enthalten (recent pin).

## Verifikation
- Unit-Test `text_correction` mit Mock (kein echtes Modell für Logik nötig).
- Lokaler Spike: echte Korrektur auf 3 deutschen Diktat-Beispielen.
- Build + App-Start + Transkription mit Korrektur an.
- Frontend-Toggle verifizieren.

## Offene Punkte
- Bundle vs On-Demand für Default 0.8B (Vorschlag: Bundle).
- 2B-Export-Existenz final verifizieren (Recherche-Rate-Limit).
- Text-only ONNX im Spike bestätigen.

---

## Quellen
- [onnx-community/Qwen3.5-0.8B-ONNX (HF)](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX)
- [onnx-community/Qwen3.5-0.8B-ONNX-OPT (HF, optimiert)](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX-OPT)
- [microsoft/onnxruntime-genai Releases (PR #2043 Qwen3.5-Support)](https://github.com/microsoft/onnxruntime-genai/releases)
- [Qwen/Qwen3.5-0.8B (HF Basis-Modell)](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Vorherige Research: Qwen3-ASR Unified](file:///home/hermes/Documents/Hermes-Notizen/Recherchen/research-qwen3-asr-unified-phase3b.md)
- [Vorherige Research: NPU Small LLM Phase 3B](file:///home/hermes/Documents/Hermes-Notizen/Recherchen/research-npu-small-llm-phase3b.md)
