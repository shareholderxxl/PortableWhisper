# Master-Prompt: Whisper4Windows Refactoring, NPU Optimization & Handy-UI Integration

## Rolle & Ziel
Du bist ein erfahrener Systems Engineer und AI Developer. Unser Ziel ist es, das kompakte Repository `Whisper4Windows` (Tauri-Frontend + Python-FastAPI-Backend) zu klonen und in eine kommerziell nutzbare, komplett portable STT-Anwendung ohne Admin-Rechte zu verwandeln. 

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

1. **Performance-Optimierung:** Um Laptops ohne NVIDIA-Grafikkarte optimal zu unterstützen, stellen wir das Backend von PyTorch auf `onnxruntime-directml` um.
2. **NPU-Ansteuerung:** Konfiguriere den DirectML-Execution-Provider (`DmlExecutionProvider`), um die NPUs von Intel (Core Ultra) und AMD (Ryzen AI) unter Windows 11 nativ und extrem stromsparend anzusprechen.
3. **Modell-Wechsel:** Ersetze das PyTorch-Modell durch die für ONNX optimierte Version `onnx-community/whisper-large-v3-turbo-german-ONNX`, um die Inferenzzeit radikal zu senken.

---

## PHASE 3: Lokale LLM-Korrekturschicht (Handy-Inspired UI)

1. **UI nach dem Vorbild von Handy:** Integriere im Tauri-Frontend unter den Einstellungen einen Tab "Post-Processing". Der Nutzer kann dort eigene, benannte Prompts erstellen (z. B. "Protokoll-Stil", "Standard-Korrektur") und den exakten System-Prompt-Text in einem Textfeld editieren.
2. **Autarke Inferenz:** Anders als bei Handy (das Ollama voraussetzt) binden wir ein ultrakleines lokales LLM (Qwen 3 oder Gemma 4 2B) direkt in unser bestehendes Python-Backend ein (Ausführung via ONNX-Runtime).
3. **Workflow:** Nach der Whisper-Transkription wird der Rohtext automatisch an die interne LLM-Engine übergeben, anhand des im UI gewählten Prompts bereinigt (Entfernen von Stotterern, Grammatikkorrektur) und erst dann in die Windows-Zwischenablage injiziert.

---

## DEINE ERSTE AUFGABE (Phase 1)

1. Analysiere das geklonte `Whisper4Windows`-Repository.
2. Zeige mir, wie wir die `tauri.conf.json` anpassen müssen, um das Python-Backend als Sidecar zu registrieren und die App als portables ZIP zu packen.
3. Erkläre mir, welche PyInstaller-Konfiguration nötig ist, um das Modell `primeline/whisper-large-v3-german` beim ersten App-Start sauber in den lokalen AppData-Ordner des Nutzers herunterzuladen.

**Gib mir zunächst nur den Umsetzungsplan für diese erste Aufgabe aus.**