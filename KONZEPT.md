# Master-Prompt: PortableWhisper Refactoring, NPU Optimization & Handy-UI Integration

> **Status (Phase 1):** In Umsetzung auf Branch `phase-1`. Details und
> Entscheidungen siehe `PHASE1_PLAN.md` und `AGENTS.md`.
> Wichtige Abweichungen vom Original-Prompt:
> - Standard-Modell bleibt `small`; `primeline/whisper-large-v3-german` ist
>   zurückgestellt (liegt im Transformers-Format vor, faster-whisper braucht
>   CTranslate2) — siehe `whisper_engine.py`.
> - Backend-Port: 8765 (statt 8000).
> - PyInstaller **ohne torch** (nur ctranslate2).
> - Build-Artefakt: nur portables ZIP (kein MSI/NSIS), gebaut via GitHub Actions.

## Rolle & Ziel
Du bist ein erfahrener Systems Engineer und AI Developer. Unser Ziel ist es, das kompakte Repository `PortableWhisper` (Tauri-Frontend + Python-FastAPI-Backend) zu klonen und in eine kommerziell nutzbare, komplett portable STT-Anwendung ohne Admin-Rechte zu verwandeln. 

Die Benutzeroberfläche für die Textkorrektur soll vom Open-Source-Projekt `cjpais/handy` inspiriert sein, die Ausführung des LLMs erfolgt jedoch – anders als bei Handy – vollkommen autark und lokal direkt in unserem eigenen, gebündelten Python-Backend (Zero-Dependency für den Endnutzer).

Wir arbeiten das Projekt strikt in drei aufeinander aufbauenden Phasen ab. Beginne heute ausschließlich mit **Phase 1**.

---

## PHASE 1: Zero-Dependency & Portabilität (Fokus: HEUTE)

1. **Ziel:** Die App muss als reines ZIP-Archiv laufen. Beim Entpacken und Starten der `.exe` dürfen keine Admin-Rechte (UAC-Prompt) und keine installierten Abhängigkeiten (kein Python, kein Node.js) benötigt werden.
2. **Standard-Modell:** Nutze für die Transkription das spezialisierte deutsche Modell `primeline/whisper-large-v3-german`.
3. **Backend-Kompilierung:** Nutze `PyInstaller` im Python-Verzeichnis, um das FastAPI-Backend mitsamt dem Modell in eine eigenständige Binärdatei (`whisper-backend.exe`) zu "freezen".
4. **Tauri-Sidecar:** Integriere diese `.exe` als Tauri-Sidecar, sodass sie beim Start der Haupt-App im Hintergrund mitgestartet und beim Schließen sauber beendet wird.
5. **Pfad-Management:** Biege alle Dateioperationen (Modellgewichte, temporäre Audio-Chunks) so um, dass sie relativ im Benutzerverzeichnis unter `%LOCALAPPDATA%/DeinApp_Name/` abgelegt werden. Niemals in geschützte Systemordner schreiben!
6. **Tauri-Konfiguration:** Stelle das Build-Target in der `tauri.conf.json` so ein, dass beim Befehl `tauri build` ein portables ZIP-Archiv erzeugt wird.

---

## PHASE 2: ONNX Runtime & Windows NPU-Beschleunigung (Mittelfristig)

> **Entwicklung:** Die Umsetzung erfolgt auf einem **separaten GitHub-Branch** (z. B. `phase-2`).  
> **Wichtig:** Phase 2 wird NICHT in `phase-1` entwickelt. Der Branch `phase-2` wird vom `main`-Branch (oder dem stabilen Stand vor Phase-1-Änderungen) abgezweigt.

1. **Performance-Optimierung:** Um Laptops ohne NVIDIA-Grafikkarte (wie z. B. mit AMD Radeon oder Intel Iris Xe) optimal zu unterstützen, stellen wir das Backend von PyTorch auf `onnxruntime-directml` um.
2. **NPU-Ansteuerung:** Konfiguriere den DirectML-Execution-Provider (`DmlExecutionProvider`), um die NPUs von Intel (Core Ultra) und AMD (Ryzen AI) unter Windows 11 nativ und extrem stromsparend anzusprechen.
3. **Modell-Wechsel (erster Schritt):** Das primäre ONNX-Modell ist `amd/whisper-large-turbo-onnx-npu`. Dieses ist speziell für AMD-NPU-Beschleunigung optimiert und enthält eine Vitis-AI-kompatible Encoder-Variante (`.rai`). Datei-Struktur:
   * `decoder_model.onnx` (691 MB)
   * `encoder_model.onnx` (2,73 MB) + `encoder_model.onnx.data` (2,55 GB)
   * Optional: `ggml-large-v3-turbo-encoder-vitisai.rai` (743 MB) für AMD-NPU
4. **UI-Geräteauswahl für volle Benutzerkontrolle:**
   * Erweitere den Einstellungsbereich in der UI unter „Processing Device“ um **vier Optionen**: **Auto**, **CPU**, **GPU** und **NPU**.
   * Die Auswahl wird wie folgt im Backend umgesetzt:
     * **Auto:** Wählt automatisch die effizienteste Hardware (NPU > GPU > CPU).
     * **CPU:** Erzwingt die Ausführung auf dem Hauptprozessor (`CPUExecutionProvider`).
     * **GPU:** Erzwingt die Ausführung auf der Grafikkarte (z. B. Radeon 860M) über DirectML.
     * **NPU:** Erzwingt die Ausführung auf der NPU (z. B. AMD Ryzen AI) über DirectML.
   * **UI-Code:** Die Geräteauswahl existiert bereits in Phase 1 als UI-Element in `index.html` (Auto/GPU/CPU). In Phase 2 wird dieses Element **wieder aktiviert** und um die Option **NPU** erweitert.
5. **Modell-Austauschbarkeit (optional / später umsetzen):**
   > **Hinweis:** Dieser Punkt ist optional und wird erst in einem späteren Schritt von Phase 2 umgesetzt. Im ersten Schritt wird ausschließlich das AMD-Modell unterstützt.
   * Der Backend-Code soll langfristig in der Lage sein, zwischen verschiedenen ONNX-Modellen zu wechseln, da das Datei-Layout variieren kann.
   * **Aktuell unterstützt:** `amd/whisper-large-turbo-onnx-npu` (flache Struktur, keine `decoder_with_past`)
   * **Optional später:** `onnx-community/whisper-large-v3-turbo-german-ONNX` (ONNX-Dateien in `onnx/`-Unterordner, inkl. `decoder_with_past_model.onnx`)
   * **Umsetzung:** Der Code muss prüfen, ob Dateien flach oder in einem `onnx/`-Unterordner liegen, und ggf. `decoder_with_past` nutzen, falls vorhanden. Der Tokenizer muss ebenfalls je nach Modell ausgewählt werden (AMD hat keinen eigenen Tokenizer).

---

## PHASE 3: Lokale LLM-Korrekturschicht (Handy-Inspired UI)

1. **UI nach dem Vorbild von Handy:** Integriere im Tauri-Frontend unter den Einstellungen einen Tab "Post-Processing". Der Nutzer kann dort eigene, benannte Prompts erstellen (z. B. "Protokoll-Stil", "Standard-Korrektur") und den exakten System-Prompt-Text in einem Textfeld editieren.
2. **Autarke Inferenz:** Anders als bei Handy (das Ollama voraussetzt) binden wir ein ultrakleines lokales LLM (Qwen 3 oder Gemma 4 2B) direkt in unser bestehendes Python-Backend ein (Ausführung via ONNX-Runtime).
3. **Workflow:** Nach der Whisper-Transkription wird der Rohtext automatisch an die interne LLM-Engine übergeben, anhand des im UI gewählten Prompts bereinigt (Entfernen von Stotterern, Grammatikkorrektur) und erst dann in die Windows-Zwischenablage injiziert.

---

## DEINE ERSTE AUFGABE (Phase 1)

1. Analysiere das geklonte `PortableWhisper`-Repository.
2. Zeige mir, wie wir die `tauri.conf.json` anpassen müssen, um das Python-Backend als Sidecar zu registrieren und die App als portables ZIP zu packen.
3. Erkläre mir, welche PyInstaller-Konfiguration nötig ist, um das Modell `primeline/whisper-large-v3-german` beim ersten App-Start sauber in den lokalen AppData-Ordner des Nutzers herunterzuladen.

**Gib mir zunächst nur den Umsetzungsplan für diese erste Aufgabe aus.**