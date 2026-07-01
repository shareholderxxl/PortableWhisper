# Whisper4Windows Simplification Plan

**Datum:** 2026-06-30  
**Ziel:** Vereinfachung der App auf ein festes Modell, keine Downloads, reduzierte UI

---

## Überblick

Die App wird drastisch vereinfacht: **Ein festes Modell** (Whisper 3 Large Turbo), **keine Downloads**, **einfache UI** (nur Settings + Lizenz), **portable Daten** (alles im App-Ordner).

---

## Benutzer-Entscheidungen

1. **Modell im Build:** NEIN - User lädt nachträglich (ZIP bleibt klein ~200 MB)
2. **Ordner-Struktur:** NEU - `data/models/default/` für Standardmodell
3. **Modell-Ersatz:** JA - User kann beliebige ähnliche Modelle ersetzen
4. **Download-Skript:** NEIN - vollständiges manuelles Modell-Management
5. **Modell-Dokumentation:** JA - detaillierte README-Sektion
6. **Backend-Health-Check:** JA - dedizierter `/health` Endpunkt
7. **Fehlerbehandlung:** JA - freundlicher Dialog mit Anleitung bei fehlendem Modell
8. **Lizenz-Seite:** Nur Links + Zusammenfassung (kein vollständiger Lizenztext)

---

## PHASE 1: Python Backend (Modell-Management)

### 1.1 path_redirect.py erweitern
**Datei:** `backend/runtime_hooks/path_redirect.py`

Änderungen:

```python
# Neue Konstante nach Zeile 45
DEFAULT_MODELS_DIR: Path = APP_DIR / "models" / "default"

# Ordner erstellen nach Zeile 50
DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Neue Funktion nach Zeile 71
def get_default_models_dir() -> Path:
    """Gibt das Verzeichnis für manuell platzierte Modelle zurück."""
    return DEFAULT_MODELS_DIR
```

---

### 1.2 whisper_engine.py umschreiben
**Datei:** `backend/whisper_engine.py`

Änderungen:

#### Neue Funktion hinzufügen (nach Zeile 177):

```python
def resolve_model_path(model: str) -> tuple[Path, bool]:
    """
    Resolve model path for "default" model name or regular models.
    
    Args:
        model: Model name or "default" for user-managed model
    
    Returns:
        Tuple of (model_path, is_local_only)
        - model_path: Path to the model directory or None for HF models
        - is_local_only: True if model must be local (no download allowed)
    """
    if model.lower() == "default":
        # Check if default directory contains a model
        default_dir = get_default_models_dir()
        
        # Look for model directory structure (CTranslate2 format)
        # Expected structure: data/models/default/model_files/
        # CTranslate2 model files: config.json, model.bin, tokenizer.json, etc.
        
        # Check if default directory contains any CTranslate2 model files
        required_files = ["config.json", "model.bin", "tokenizer.json"]
        
        if all((default_dir / f).exists() for f in required_files):
            logger.info(f"✅ Using local default model from: {default_dir}")
            return default_dir, True
        else:
            logger.warning(f"⚠️ Default model requested but required files missing in {default_dir}")
            logger.warning(f"   Required files: {required_files}")
            logger.warning(f"   Available files: {list(default_dir.iterdir())}")
            return None, True
    else:
        # Regular model - use HF cache path (no auto-download allowed)
        return None, False
```

#### Funktion get_default_models_dir() hinzufügen (nach Zeile 177):

```python
def get_default_models_dir() -> Path:
    """Get the default models directory for user-managed models."""
    from runtime_hooks.path_redirect import get_default_models_dir
    return get_default_models_dir()
```

#### is_model_downloaded() ändern (Zeilen 256-288):

```python
def is_model_downloaded(self, model_size: str = None) -> bool:
    """
    Check if a model is already downloaded

    Args:
        model_size: Model alias (tiny/base/small/medium/large-v3/large-v3-turbo)
                    ODER "default" für manuell platziertes Modell
                    ODER eine vollständige HF-Repo-ID der Form 'org/repo'.
                    (uses self.model_size if None)

    Returns:
        True if model is downloaded, False otherwise
    """
    if model_size is None:
        model_size = self.model_size

    # Handle "default" model
    if model_size.lower() == "default":
        default_dir = get_default_models_dir()
        required_files = ["config.json", "model.bin", "tokenizer.json"]
        
        if all((default_dir / f).exists() for f in required_files):
            logger.info(f"✅ Default model is available in: {default_dir}")
            return True
        else:
            logger.warning(f"⚠️ Default model not found in: {default_dir}")
            return False

    # Handle regular models - check HF cache
    models_dir = get_models_dir()
    dir_name = _model_cache_dir_name(model_size)

    # Check direct path and hub subdirectory path for compatibility
    paths_to_check = [
        models_dir / dir_name,
        models_dir / "hub" / dir_name
    ]

    for model_path in paths_to_check:
        if model_path.exists():
            snapshot_dir = model_path / "snapshots"
            if snapshot_dir.exists() and any(snapshot_dir.iterdir()):
                logger.info(f"✅ Model '{model_size}' is already downloaded ({model_path})")
                return True

    logger.warning(f"⚠️ Model '{model_size}' is not downloaded (and auto-download is disabled)")
    return False
```

#### load_model() umschreiben (Zeilen 290-372):

```python
def load_model(self) -> bool:
    """
    Load the Whisper model with automatic GPU compute type fallback, then CPU fallback
    Does NOT auto-download models - they must be present locally.

    Returns:
        True if successful, False otherwise
    """
    if not WHISPER_AVAILABLE:
        logger.error("❌ faster-whisper is not installed!")
        return False

    if self.is_loaded:
        logger.info("Model already loaded")
        return True

    try:
        logger.info(f"📥 Loading Whisper model: {self.model_size}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Compute type: {self.compute_type}")

        # Resolve model path for "default" or regular models
        model_path, is_local_only = resolve_model_path(self.model_size)

        # For "default" model, load from default directory
        if is_local_only and model_path is not None:
            if not model_path.exists():
                logger.error(f"❌ Default model directory not found: {model_path}")
                return False

            try:
                logger.info(f"🔄 Loading local default model from: {model_path}")
                self.model = WhisperModel(
                    str(model_path),  # Path to local model directory
                    device=self.device,
                    compute_type=self.compute_type
                    # NO download_root - local model only
                )

                self.is_loaded = True
                logger.info(f"✅ Default model loaded successfully on {self.device.upper()}")
                return True

            except Exception as e:
                logger.error(f"❌ Failed to load default model: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False

        elif is_local_only and model_path is None:
            logger.error("❌ Default model requested but not found in data/models/default/")
            logger.error("   Please place CTranslate2 model files in the default directory")
            logger.error("   Required files: config.json, model.bin, tokenizer.json")
            return False

        # For regular models, ensure they're downloaded (no auto-download)
        if not self.is_model_downloaded():
            logger.error(f"❌ Model '{self.model_size}' is not downloaded and auto-download is disabled")
            logger.error("   Please download the model manually or place it in data/models/default/")
            return False

        # Create models directory if it doesn't exist (for HF cache compatibility)
        models_dir = get_models_dir()

        # If using CUDA, try compute types in order of efficiency
        if self.device == "cuda":
            compute_types_to_try = self._get_cuda_compute_type_fallbacks()

            # If user specified a specific compute type, try that first
            if self.compute_type != "auto" and self.compute_type not in compute_types_to_try:
                compute_types_to_try.insert(0, self.compute_type)

            for compute_type in compute_types_to_try:
                try:
                    logger.info(f"🔄 Trying CUDA with compute type: {compute_type}")
                    self.model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=compute_type,
                        download_root=str(models_dir)  # For cache compatibility only
                    )

                    # Success! Cache this compute type
                    self.compute_type = compute_type
                    self.is_loaded = True
                    logger.info(f"✅ Model loaded successfully on CUDA with {compute_type}: {self.model_size}")
                    return True

                except Exception as compute_error:
                    logger.warning(f"⚠️ CUDA with {compute_type} failed: {compute_error}")
                    # Continue to next compute type
                    continue

            # All CUDA compute types failed, fall back to CPU
            logger.warning("⚠️ All CUDA compute types failed")
            logger.info("🔄 Falling back to CPU...")
            self.device = "cpu"
            self.compute_type = "int8"

        # Try loading with current device/compute_type (either CPU from start, or CPU fallback)
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(models_dir)  # For cache compatibility only
            )

            self.is_loaded = True
            if self._cuda_detected and self.device == "cpu":
                logger.info(f"✅ Model loaded successfully on CPU (GPU fallback): {self.model_size}")
            else:
                logger.info(f"✅ Model loaded successfully on {self.device.upper()}: {self.model_size}")
            return True

        except Exception as final_error:
            logger.error(f"❌ Final loading attempt failed: {final_error}")
            raise

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
```

---

### 1.3 main.py Health-Check hinzufügen
**Datei:** `backend/main.py`

Änderungen:

#### Neuer /health Endpunkt (nach Zeile 180):

```python
@app.get("/health")
async def health_check():
    """Health check endpoint with model and device status."""
    global whisper_engine
    
    model_status = "missing"
    if whisper_engine is not None and whisper_engine.is_loaded:
        model_status = "loaded"
    elif whisper_engine is not None and whisper_engine.is_model_downloaded():
        model_status = "available"
    
    device = "unknown"
    if whisper_engine is not None:
        device = str(whisper_engine.device)
    
    return {
        "status": "ok",
        "backend": "whisper-backend",
        "model": "default",
        "model_status": model_status,
        "device": device,
        "version": "1.0.0"
    }
```

#### /load_model Statusmeldung ändern (Zeile 259):

```python
# ALT:
model_loading_info = {
    "model": whisper_engine.model_size,
    "status": "Downloading model..." if not whisper_engine.is_model_downloaded() else "Loading model..."
}

# NEU:
model_loading_info = {
    "model": whisper_engine.model_size,
    "status": "Loading model..."  # No auto-download
}
```

#### /load_model Fehlermeldung erweitern (Zeilen 269-273):

```python
# ALT:
if not success:
    return {
        "status": "error",
        "message": "Failed to load Whisper model"
    }

# NEU:
if not success:
    error_msg = "Failed to load Whisper model."
    if whisper_engine.model_size.lower() == "default" and not whisper_engine.is_model_downloaded():
        error_msg += " Please place CTranslate2 model files in data/models/default/"
    
    return {
        "status": "error",
        "message": error_msg,
        "details": "Auto-download is disabled. Place models in data/models/default/ or download manually."
    }
```

---

## PHASE 2: Rust Backend Vereinfachung

### 2.1 lib.rs Commands entfernen
**Datei:** `frontend/src-tauri/src/lib.rs`

Entfernen:
- `download_model` command (Zeilen 583-623)
- `set_model_and_device` command (Zeilen 517-581)
- `selected_model` aus AppState (Zeilen ~66-67, ~72-74, ~121-124)

Ändern:

#### Settings Struktur (Zeilen 25-51):

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Settings {
    // pub selected_model: String,  // ENTFERNEN
    pub selected_device: String,
    pub selected_microphone: Option<i32>,
    pub use_clipboard: bool,
    pub selected_language: String,
    pub toggle_shortcut: String,
    pub cancel_shortcut: String,
    pub recording_mode: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            // selected_model: "tiny".to_string(),  // ENTFERNEN
            selected_device: "auto".to_string(),
            selected_microphone: None,
            use_clipboard: true,
            selected_language: "auto".to_string(),
            toggle_shortcut: "Ctrl+\\".to_string(),
            cancel_shortcut: "Escape".to_string(),
            recording_mode: "toggle".to_string(),
        }
    }
}
```

#### AppState (Zeilen 55-84):

```rust
pub struct AppState {
    // pub selected_model: Arc<Mutex<String>>,  // ENTFERNEN
    pub selected_device: Arc<Mutex<String>>,
    pub selected_microphone: Arc<Mutex<Option<i32>>>,
    pub use_clipboard: Arc<Mutex<bool>>,
    pub selected_language: Arc<Mutex<String>>,
    pub toggle_shortcut: Arc<Mutex<String>>,
    pub cancel_shortcut: Arc<Mutex<String>>,
    pub recording_mode: Arc<Mutex<String>>,
    pub settings_path: PathBuf,
    pub backend_child: Arc<Mutex<Option<Child>>>,
}

impl AppState {
    pub fn new(settings_path: PathBuf) -> Self {
        Self {
            // selected_model: Arc::new(Mutex::new("default".to_string())),  // ENTFERNEN
            selected_device: Arc::new(Mutex::new("auto".to_string())),
            selected_microphone: Arc::new(Mutex::new(None)),
            use_clipboard: Arc::new(Mutex::new(true)),
            selected_language: Arc::new(Mutex::new("auto".to_string())),
            toggle_shortcut: Arc::new(Mutex::new("Ctrl+\\".to_string())),
            cancel_shortcut: Arc::new(Mutex::new("Escape".to_string())),
            recording_mode: Arc::new(Mutex::new("toggle".to_string())),
            settings_path,
            backend_child: Arc::new(Mutex::new(None)),
        }
    }
}
```

#### save_settings() Funktion (Zeilen 107-126):

```rust
async fn save_settings(&self) {
    let settings = Settings {
        // selected_model: self.selected_model.lock().await.clone(),  // ENTFERNEN
        selected_device: self.selected_device.lock().await.clone(),
        selected_microphone: *self.selected_microphone.lock().await,
        use_clipboard: *self.use_clipboard.lock().await,
        selected_language: self.selected_language.lock().await.clone(),
        toggle_shortcut: self.toggle_shortcut.lock().await.clone(),
        cancel_shortcut: self.cancel_shortcut.lock().await.clone(),
        recording_mode: self.recording_mode.lock().await.clone(),
    };

    if let Ok(json) = serde_json::to_string_pretty(&settings) {
        if let Err(e) = fs::write(&self.settings_path, json) {
            log::error!("❌ Failed to save settings: {}", e);
        } else {
            log::info!("💾 Settings saved to {:?}", self.settings_path);
        }
    }
}
```

#### load_settings() Funktion (Zeilen 86-105):

```python
async fn load_settings(&self) -> Settings {
    match fs::read_to_string(&self.settings_path) {
        Ok(content) => {
            match serde_json::from_str::<Settings>(&content) {
                Ok(settings) => {
                    log::info!("✅ Loaded settings from {:?}", self.settings_path);
                    
                    // Update state from loaded settings
                    *self.selected_device.lock().await = settings.selected_device;
                    *self.selected_microphone.lock().await = settings.selected_microphone;
                    *self.use_clipboard.lock().await = settings.use_clipboard;
                    *self.selected_language.lock().await = settings.selected_language;
                    *self.toggle_shortcut.lock().await = settings.toggle_shortcut;
                    *self.cancel_shortcut.lock().await = settings.cancel_shortcut;
                    *self.recording_mode.lock().await = settings.recording_mode;
                    
                    settings
                }
                Err(e) => {
                    log::warn!("⚠️ Failed to parse settings: {}, using defaults", e);
                    Settings::default()
                }
            }
        }
        Err(_) => {
            log::info!("📝 No settings file found, using defaults");
            Settings::default()
        }
    }
}
```

#### start_recording() Funktion (Zeilen 308-349):

```rust
// ALT:
let model = state.selected_model.lock().await.clone();

// NEU:
let model = "default".to_string();  // Feste Modell-Auswahl
```

#### Startup-Logik entfernen (Zeilen 1395-1432):

Komplett entfernen - kein automatisches Laden des Modells beim Startup mehr. Das Modell wird erst beim ersten Recording geladen.

#### invoke_handler aktualisieren (Zeilen 1446-1447):

```rust
// ALT:
download_model,
set_model_and_device,

// NEU: Beide ENTFERNEN
```

---

## PHASE 3: UI-Vereinfachung

### 3.1 index.html Struktur
**Datei:** `frontend/dist/index.html`

Entfernen (Zeilen):
- Model Quality Dropdown (784-799)
- Processing Device Buttons (803-813)
- GPU Libraries Install (815-823)
- Model Management Section (857-866)
- Update Application (894-902)
- `MODELS_METADATA` (1569-1578)
- `checkDownloadedModels()` (1584-1618)
- `renderModelManagementList()` (1620-1677)
- `startModelDownload()` (1679-1710)
- `startModelPolling()` (1712-1769)
- `checkForUpdates()` (2005-2017)

Hinzufügen:

#### Vereinfachte Sidebar (Zeilen 674-689):

```html
<div class="sidebar">
    <div class="sidebar-item active" onclick="navigateToPage('settings')">
        <span class="sidebar-item-icon">⚙️</span>
        Settings
    </div>
    
    <div class="sidebar-item" onclick="navigateToPage('licenses')">
        <span class="sidebar-item-icon">📜</span>
        Licenses
    </div>
</div>
```

#### Settings Page umstrukturieren (Zeilen 724-931):

Neue Struktur:
- Keyboard Shortcuts (behalten)
- Backend Status (behalten + erweitern)
- Language Selection (behalten)
- **Model Status** (neu, fest)
- Appearance (behalten)
- Sound (separate Seite, behalten)

#### Licenses Page hinzufügen (nach Sound Page, ab Zeile ~994):

```html
<div id="page-licenses" class="page hidden">
    <div class="main-content">
        <div class="section-title">Lizenzen</div>
        
        <div class="config-section">
            <h3 class="config-section-title">Whisper4Windows</h3>
            <div class="config-row-description">
                <p><strong>Lizenz:</strong> MIT License</p>
                <p><strong>Fork attribution:</strong> Original project by Bader Aljabri.</p>
                <p><strong>GitHub:</strong> <a href="https://github.com/shareholderxxl/Whisper4Windows" target="_blank">https://github.com/shareholderxxl/Whisper4Windows</a></p>
                <p><strong>Vollständiger Lizenztext:</strong> Siehe <code>LICENSE</code> Datei</p>
            </div>
        </div>
        
        <div class="config-section">
            <h3 class="config-section-title">Whisper 3 Large Turbo</h3>
            <div class="config-row-description">
                <p><strong>Source:</strong> Systran/faster-whisper-large-v3-turbo</p>
                <p><strong>Lizenz:</strong> MIT License</p>
                <p><strong>Model weights:</strong> <a href="https://huggingface.co/Systran/faster-whisper-large-v3-turbo" target="_blank">https://huggingface.co/Systran/faster-whisper-large-v3-turbo</a></p>
            </div>
        </div>

        <div class="config-section">
            <h3 class="config-section-title">Abhängigkeiten</h3>
            <div class="config-row-description">
                <p><strong>faster-whisper:</strong> MIT License - <a href="https://github.com/SYSTRAN/faster-whisper" target="_blank">GitHub</a></p>
                <p><strong>CTranslate2:</strong> BSD-3-Clause - <a href="https://github.com/OpenNMT/CTranslate2" target="_blank">GitHub</a></p>
                <p><strong>Tauri 2:</strong> MIT/Apache-2.0 - <a href="https://tauri.app" target="_blank">tauri.app</a></p>
                <p><strong>PyTorch</strong> (nur im Build-Prozess): BSD-3-Clause - <a href="https://pytorch.org" target="_blank">pytorch.org</a></p>
            </div>
        </div>
    </div>
</div>
```

#### Model Status Anzeige (in Settings Page, nach Backend Status):

```html
<div class="config-row">
    <div class="config-row-left">
        <div class="config-row-title">Aktives Modell</div>
        <div class="config-row-description">Whisper 3 Large Turbo (Standard)</div>
    </div>
    <div class="config-row-right">
        <span class="status-indicator" id="modelStatusIndicator">Prüfe...</span>
    </div>
</div>
```

---

### 3.2 JavaScript Änderungen

Entfernen:
- `MODELS_METADATA` array (Zeilen 1569-1578)
- `checkDownloadedModels()` (Zeilen 1584-1618)
- `renderModelManagementList()` (Zeilen 1620-1677)
- `startModelDownload()` (Zeilen 1679-1710)
- `startModelPolling()` (Zeilen 1712-1769)
- `checkForUpdates()` (Zeilen 2005-2017)
- `updateModelQualityDropdown()` (falls vorhanden)
- `toggleModelDropdown()`
- `selectModel()`
- `filterModels()`

Hinzufügen/Ändern:

#### navigateToPage() aktualisieren (Zeilen 1085-1105):

```javascript
function navigateToPage(pageName, section) {
    document.querySelectorAll('.page').forEach(page => {
        page.classList.add('hidden');
    });
    
    // Neue Seiten-Namen
    if (pageName === 'settings') {
        document.getElementById('page-configuration').classList.remove('hidden');
    } else if (pageName === 'licenses') {
        document.getElementById('page-licenses').classList.remove('hidden');
    }
    
    // Sidebar active state aktualisieren
    document.querySelectorAll('.sidebar-item').forEach(item => {
        item.classList.remove('active');
    });
    event.currentTarget.classList.add('active');
    
    if (section) {
        const sectionEl = document.getElementById(`section-${section}`);
        if (sectionEl) {
            sectionEl.scrollIntoView({ behavior: 'smooth' });
        }
    }
}
```

#### checkBackendHealth() erweitern (Zeilen 1208-1236):

```javascript
async function checkBackendHealth() {
    try {
        const response = await fetch(`${BACKEND_URL}/health`, {
            method: 'GET',
            mode: 'cors',
            headers: { 'Accept': 'application/json' },
            signal: AbortSignal.timeout(3000)
        });
        const data = await response.json();

        const statusEl = document.getElementById('backendStatus');
        
        // Backend Status anzeigen
        let statusText = `${data.backend.toUpperCase()} Ready`;
        if (data.model_status === 'loaded') {
            statusText += ` (${data.model} on ${data.device})`;
        } else if (data.model_status === 'missing') {
            statusText += ` (Modell fehlt!)`;
        }
        
        statusEl.innerHTML = `<span>✅</span><span>${statusText}</span>`;
        statusEl.classList.remove('offline', 'starting', 'checking');
        lastBackendStatus = 'online';

        // Model Status Indikator aktualisieren
        const modelStatusEl = document.getElementById('modelStatusIndicator');
        if (modelStatusEl) {
            if (data.model_status === 'loaded') {
                modelStatusEl.textContent = 'Geladen';
                modelStatusEl.className = 'status-indicator online';
            } else if (data.model_status === 'available') {
                modelStatusEl.textContent = 'Verfügbar';
                modelStatusEl.className = 'status-indicator online';
            } else if (data.model_status === 'missing') {
                modelStatusEl.textContent = 'Fehlt';
                modelStatusEl.className = 'status-indicator offline';
            }
        }
    } catch (error) {
        const statusEl = document.getElementById('backendStatus');
        if (lastBackendStatus !== 'offline') {
            console.log('⚠️ Backend offline:', error.message);
        }
        statusEl.innerHTML = `<span>❌</span><span>Backend Offline (click to restart)</span>`;
        statusEl.classList.remove('starting', 'checking');
        statusEl.classList.add('offline');
        lastBackendStatus = 'offline';
    }
}
```

#### Neue Funktion: checkModelAvailability()

```javascript
async function checkModelAvailability() {
    try {
        const response = await fetch(`${BACKEND_URL}/health`);
        const data = await response.json();
        
        if (data.model_status === 'missing') {
            // Nur einmalig zeigen (session storage)
            if (!sessionStorage.getItem('missingModelShown')) {
                showMissingModelDialog();
                sessionStorage.setItem('missingModelShown', 'true');
            }
        }
    } catch (error) {
        console.error("Health check failed:", error);
    }
}
```

#### Neue Funktion: showMissingModelDialog()

```javascript
function showMissingModelDialog() {
    // Prüfen ob Dialog bereits existiert
    if (document.querySelector('.missing-model-dialog')) {
        return;
    }

    const dialog = document.createElement('div');
    dialog.className = 'missing-model-dialog';
    dialog.innerHTML = `
        <div class="dialog-content">
            <h2>❌ Modell nicht gefunden</h2>
            <p>Whisper4Windows benötigt ein Whisper-Modell, aber <code>data/models/default/</code> ist leer oder fehlt.</p>
            
            <h3>So installieren Sie das Modell:</h3>
            
            <div class="install-method">
                <strong>Option A: HuggingFace CLI (empfohlen)</strong>
                <pre>pip install huggingface-hub
huggingface-cli download Systran/faster-whisper-large-v3-turbo --local-dir data/models/default/ --local-dir-use-symlinks False</pre>
            </div>
            
            <div class="install-method">
                <strong>Option B: Manuelles Download</strong>
                <ol>
                    <li>Navigieren zu: <a href="https://huggingface.co/Systran/faster-whisper-large-v3-turbo" target="_blank">https://huggingface.co/Systran/faster-whisper-large-v3-turbo</a></li>
                    <li>Alle Dateien downloaden (config.json, model.bin, tokenizer.json, etc.)</li>
                    <li>In <code>data/models/default/</code> platzieren</li>
                </ol>
            </div>
            
            <h3>Struktur überprüfen:</h3>
            <p>Nach der Installation sollte <code>data/models/default/</code> folgende Dateien enthalten:</p>
            <ul>
                <li>config.json</li>
                <li>model.bin</li>
                <li>tokenizer.json</li>
                <li>vocabulary.txt</li>
                <li>(und weitere Dateien je nach Modell)</li>
            </ul>
            
            <h3>Weitere Informationen:</h3>
            <p>Detaillierte Anleitung siehe <strong>Lizenzen</strong> → README</p>
            <p><strong>Hinweis:</strong> Sie können auch ein anderes Whisper-Modell verwenden (z.B. Whisper Small oder Large V3 Turbo German), solange es im CTranslate2-Format vorliegt.</p>
            
            <button class="close-dialog-btn">Schließen</button>
        </div>
    `;
    
    document.body.appendChild(dialog);
    
    // Close button handler
    dialog.querySelector('.close-dialog-btn').onclick = () => {
        dialog.remove();
    };
    
    // Close on outside click
    dialog.onclick = (e) => {
        if (e.target === dialog) {
            dialog.remove();
        }
    };
}
```

#### Startup-Logik aktualisieren (am Ende des Scripts):

```javascript
// Beim Startup prüfen
window.addEventListener('DOMContentLoaded', () => {
    // Backend health polling starten
    startBackendHealthPolling();
    
    // Nach 3 Sekunden Modell-Verfügbarkeit prüfen
    setTimeout(checkModelAvailability, 3000);
});
```

#### CSS hinzufügen (in Style-Block):

```css
/* Missing Model Dialog */
.missing-model-dialog {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10000;
}

.dialog-content {
    background: var(--bg-card);
    padding: 30px;
    border-radius: 12px;
    max-width: 600px;
    max-height: 80vh;
    overflow-y: auto;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}

.dialog-content h2 {
    color: #e74c3c;
    margin-bottom: 15px;
    font-size: 20px;
}

.dialog-content h3 {
    margin: 20px 0 10px 0;
    font-size: 16px;
    color: var(--text-primary);
}

.dialog-content p {
    color: var(--text-secondary);
    line-height: 1.5;
    margin-bottom: 10px;
}

.dialog-content ul, .dialog-content ol {
    margin: 10px 0;
    padding-left: 20px;
    color: var(--text-secondary);
}

.dialog-content li {
    margin: 5px 0;
}

.dialog-content pre {
    background: rgba(0,0,0,0.1);
    padding: 12px;
    border-radius: 6px;
    margin: 10px 0;
    overflow-x: auto;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    color: var(--text-primary);
}

.dialog-content code {
    background: rgba(0,0,0,0.1);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
}

.dialog-content a {
    color: #667eea;
    text-decoration: none;
}

.dialog-content a:hover {
    text-decoration: underline;
}

.install-method {
    background: rgba(0,0,0,0.03);
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}

.close-dialog-btn {
    margin-top: 20px;
    padding: 10px 20px;
    background: var(--primary-color, #667eea);
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
}

.close-dialog-btn:hover {
    opacity: 0.9;
}

/* Model Status Indicator */
.status-indicator {
    display: inline-flex;
    align-items: center;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
}

.status-indicator.online {
    background: rgba(52, 211, 153, 0.1);
    color: #34d399;
}

.status-indicator.offline {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
}
```

---

## PHASE 4: GitHub Actions

### 4.1 build-windows.yml anpassen
**Datei:** `.github/workflows/build-windows.yml`

Entfernen (Zeilen 153-160):

```yaml
# ENTFERNEN: Pre-download default tiny model
# Set-Content -Path _predl_tiny.py -Value "import sys"
# Add-Content -Path _predl_tiny.py -Value "from faster_whisper import WhisperModel"
# Add-Content -Path _predl_tiny.py -Value "WhisperModel('tiny', download_root=sys.argv[1])"
# Write-Host "Pre-downloading default tiny model into portable data folder..."
# python _predl_tiny.py "$stage/data/models"
```

Ändern (Zeile 151):

```yaml
# ALT: New-Item -ItemType Directory -Force -Path "$stage/data/models" | Out-Null

# NEU: Ordner-Struktur anlegen (inkl. default, temp, logs)
New-Item -ItemType Directory -Force -Path "$stage/data/models/default" | Out-Null
New-Item -ItemType Directory -Force -Path "$stage/data/temp" | Out-Null
New-Item -ItemType Directory -Force -Path "$stage/data/logs" | Out-Null
```

Hinzufügen (nach Zeile 167):

```yaml
# Kopiere LIZENZ-Dateien in ZIP
Copy-Item LICENSE $stage/ -ErrorAction SilentlyContinue
Copy-Item LICENSES.md $stage/ -ErrorAction SilentlyContinue
```

---

## PHASE 5: Dokumentation

### 5.1 README.md erweitern
**Datei:** `README.md`

Am Ende hinzufügen:

```markdown
## Modell-Installation

Whisper4Windows wird **ohne vorinstalliertes Modell** ausgeliefert, um die Download-Größe gering zu halten. Sie müssen das Modell manuell downloaden und installieren.

### Standardmodell: Whisper 3 Large Turbo

**Empfohlenes Modell:** `Systran/faster-whisper-large-v3-turbo`

Dieses Modell bietet eine hervorragende Kombination aus Genauigkeit und Geschwindigkeit (Large V3 Qualität mit Turbo-Performance).

#### Installation

**Option A: HuggingFace CLI (empfohlen)**

```bash
# 1. Installieren Sie huggingface-cli
pip install huggingface-hub

# 2. Navigieren Sie zum App-Ordner
cd C:\Pfad\zu\Whisper4Windows

# 3. Laden Sie das Modell herunter
huggingface-cli download Systran/faster-whisper-large-v3-turbo --local-dir data/models/default/ --local-dir-use-symlinks False
```

**Option B: Manuelles Download**

1. Navigieren Sie zu: https://huggingface.co/Systran/faster-whisper-large-v3-turbo
2. Laden Sie alle Dateien herunter:
   - config.json
   - model.bin
   - tokenizer.json
   - vocabulary.txt
   - (und alle weiteren Dateien im Repository)
3. Platzieren Sie die Dateien im Ordner: `data/models/default/`

#### Struktur überprüfen

Nach der Installation sollte `data/models/default/` folgende Dateien enthalten:

```
data/models/default/
├── config.json
├── model.bin
├── tokenizer.json
├── vocabulary.txt
└── (weitere Dateien je nach Modell)
```

#### App starten

Starten Sie `Whisper4Windows.exe`. Das Backend lädt das Modell automatisch aus `data/models/default/`.

### Modell ersetzen

Sie können das Modell jederzeit durch ein anderes Whisper-Modell ersetzen:

1. Schließen Sie Whisper4Windows
2. Ersetzen Sie die Dateien in `data/models/default/`
3. Starten Sie die App neu

**Kompatible Modelle:**

Alle Whisper-Modelle im **CTranslate2-Format** sind kompatibel:

- Whisper Tiny: `Systran/faster-whisper-tiny` (~75 MB, sehr schnell, geringere Genauigkeit)
- Whisper Base: `Systran/faster-whisper-base` (~140 MB, schnell)
- Whisper Small: `Systran/faster-whisper-small` (~460 MB, empfohlen)
- Whisper Medium: `Systran/faster-whisper-medium` (~1.5 GB, hohe Qualität)
- Whisper Large V3: `Systran/faster-whisper-large-v3` (~3.0 GB, maximale Qualität)
- Whisper Large V3 Turbo: `Systran/faster-whisper-large-v3-turbo` (~1.6 GB, bestes Verhältnis)
- German Fine-tuned: `Reality-Interface/whisper-large-v3-german-faster-whisper` (~3.0 GB, bestes für Deutsch)

**WICHTIG:** Das Modell muss im CTranslate2-Format vorliegen (nicht im transformers-Format). Alle `faster-whisper` Modelle von Systran sind im richtigen Format.

### Fehlerbehandlung

Wenn Sie beim Start die Fehlermeldung "Modell nicht gefunden" erhalten:

1. Stellen Sie sicher, dass `data/models/default/` existiert
2. Überprüfen Sie, dass alle erforderlichen Dateien vorhanden sind
3. Prüfen Sie `data/logs/whisper-backend.log` für detaillierte Fehlerinformationen
4. Stellen Sie sicher, dass das Modell im CTranslate2-Format vorliegt

### Performance-Tipps

- **CPU-Only:** Whisper Small oder Base sind am besten für CPU-only
- **Mit GPU:** Large V3 Turbo bietet das beste Verhältnis aus Qualität und Geschwindigkeit
- **Deutsches Diktat:** German Fine-tuned Modell bietet die beste Genauigkeit für Deutsch
- **Schnelles Diktat:** Tiny ist sehr schnell, aber weniger genau

### Modell-Quellen

- Offizielle Whisper Modelle: https://huggingface.co/Systran
- German Fine-tuned: https://huggingface.co/Reality-Interface/whisper-large-v3-german-faster-whisper
- Weitere Modelle: https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&library=faster-whisper
```

### 5.2 LICENSES.md erstellen
**Datei:** `LICENSES.md` (neu)

```markdown
# Whisper4Windows - Lizenzinformationen

## Whisper4Windows

**Lizenz:** MIT License

Copyright (c) 2024 ShareholderXXL

Basiert auf dem originalen Projekt von Bader Aljabri.

Vollständiger Lizenztext siehe `LICENSE` Datei in diesem Repository.

## Whisper 3 Large Turbo Modell

**Source:** Systran/faster-whisper-large-v3-turbo
**Lizenz:** MIT License
**URL:** https://huggingface.co/Systran/faster-whisper-large-v3-turbo

Das Modell steht unter der MIT License zur Verfügung.

## Abhängigkeiten

### Backend (Python)

- **faster-whisper:** MIT License - https://github.com/SYSTRAN/faster-whisper
- **CTranslate2:** BSD-3-Clause - https://github.com/OpenNMT/CTranslate2
- **fastapi:** MIT License - https://github.com/tiangolo/fastapi
- **uvicorn:** BSD-3-Clause - https://github.com/encode/uvicorn
- **numpy:** BSD-3-Clause - https://numpy.org
- **soundfile:** BSD-3-Clause - https://github.com/bastibe/python-soundfile
- **scipy:** BSD-3-Clause - https://scipy.org
- **huggingface-hub:** Apache-2.0 - https://github.com/huggingface/huggingface_hub

### Frontend (Tauri)

- **Tauri 2:** MIT/Apache-2.0 - https://tauri.app
- **PyO3:** Apache-2.0 - https://github.com/PyO3/pyo3
- **tokio:** MIT - https://github.com/tokio-rs/tokio
- **reqwest:** MIT/Apache-2.0 - https://github.com/seanmonstar/reqwest

### Build-Tools (nur im Build-Prozess)

- **PyTorch** (nur im Build-Prozess): BSD-3-Clause - https://pytorch.org
- **Rust/Cargo:** Apache-2.0/MIT - https://www.rust-lang.org

## Danksagung

Danke an:
- OpenAI für das Whisper-Modell und die ursprüngliche Idee
- Systran für faster-whisper und CTranslate2
- Bader Aljabri für das ursprüngliche Projekt (whisper4windows)
- Der Whisper-Community für Beiträge und Verbesserungen
- Allen Mitwirkenden an den verwendeten Open-Source-Projekten

## Lizenz-Kompatibilität

Diese Anwendung verwendet Bibliotheken unter verschiedenen Open-Source-Lizenzen (MIT, BSD-3-Clause, Apache-2.0). Alle verwendeten Lizenzen sind kompatibel mit der MIT License unter der Whisper4Windows veröffentlicht wird.

## Hinweis zur Modell-Nutzung

Die verwendeten Whisper-Modelle stehen unter der MIT License und können frei verwendet werden. Bitte beachten Sie jedoch:
- Die Modelle wurden von OpenAI trainiert
- Verwenden Sie die Modelle ethisch und verantwortungsbewusst
- Beachten Sie die Nutzungsbedingungen von OpenAI für AI-Modelle

Für mehr Informationen siehe: https://openai.com/policies/usage-policies
```

---

## PHASE 6: Validierung & Testing

### 6.1 Backend-Testing

```python
# Test 1: default Modell-Ordner wird erstellt
from runtime_hooks.path_redirect import get_default_models_dir
default_dir = get_default_models_dir()
assert default_dir.exists()
assert str(default_dir).endswith("data/models/default")

# Test 2: resolve_model_path()
from whisper_engine import resolve_model_path
model_path, is_local = resolve_model_path("default")
if default_files_present:
    assert model_path == default_dir
    assert is_local == True
else:
    assert model_path is None
    assert is_local == True

# Test 3: is_model_downloaded("default")
engine = WhisperEngine(model_size="default")
assert engine.is_model_downloaded() == bool(default_files_present)

# Test 4: load_model() mit default
if default_files_present:
    success = engine.load_model()
    assert success == True
    assert engine.is_loaded == True
else:
    success = engine.load_model()
    assert success == False
    assert engine.is_loaded == False

# Test 5: /health Endpunkt
import requests
response = requests.get("http://127.0.0.1:8765/health")
data = response.json()
assert data["status"] == "ok"
assert data["model"] == "default"
assert data["model_status"] in ["loaded", "available", "missing"]
```

### 6.2 Frontend-Testing

- ✅ UI zeigt nur Settings + Licenses Seiten
- ✅ Keine Modell-Download-Buttons
- ✅ Keine Modell-Auswahl-Dropdown
- ✅ Keine Update-Funktionalität
- ✅ Modell-Status zeigt "Whisper 3 Large Turbo (Standard)"
- ✅ /health Endpunkt wird aufgerufen und Status angezeigt
- ✅ Freundlicher Fehlerdialog bei fehlendem Modell
- ✅ Spracheauswahl funktioniert
- ✅ Hotkeys funktionieren
- ✅ Sound-Einstellungen funktionieren

### 6.3 Integration-Testing

1. **ZIP entpacken → leeres data/models/default/**
   - App starten
   - Fehlerdialog wird angezeigt
   - Backend-Status zeigt "Modell fehlt!"

2. **Modell platzieren**
   - Modell in data/models/default/ kopieren
   - App neu starten
   - Backend-Status zeigt "Modell geladen"
   - Recording funktioniert

3. **Modell ersetzen**
   - App schließen
   - Dateien in data/models/default/ ersetzen
   - App neu starten
   - Backend lädt neues Modell
   - Recording mit neuem Modell

---

## Zusammenfassung der Änderungen

| Phase | Dateien | Zeilen (ca.) | Priorität |
|-------|---------|--------------|-----------|
| 1: Python Backend | path_redirect.py, whisper_engine.py, main.py | +180, -50 | Hoch |
| 2: Rust Backend | lib.rs | -120, +30 | Hoch |
| 3: UI | index.html | -380, +120 | Hoch |
| 4: GitHub Actions | build-windows.yml | -10, +15 | Mittel |
| 5: Dokumentation | README.md, LICENSES.md | +150 | Mittel |
| 6: Testing | - | - | Hoch |

**Gesamt:** ~560 Zeilen entfernt, ~495 Zeilen hinzugefügt

**ZIP-Größe:** ~200 MB (statt ~270 MB mit tiny-Modell)

---

## User-Workflow

### Erstes Setup

1. User lädt ZIP herunter (~200 MB)
2. Entpackt ZIP in beliebigen Ordner
3. Startet `Whisper4Windows.exe`
4. App zeigt Fehlerdialog: "Modell nicht gefunden"
5. User folgt Anleitung im Dialog:
   - Option A: huggingface-cli download
   - Option B: Manuelles Download
6. User platziert Modell in `data/models/default/`
7. App neu starten
8. App funktioniert

### Modell-Ersatz

User kann Modell jederzeit ersetzen:
1. App schließen
2. Dateien in `data/models/default/` ersetzen
3. App neu starten
4. Backend lädt neues Modell automatisch

---

## Risiken & Mitigation

| Risiko | Wahrscheinlichkeit | Auswirkung | Mitigation |
|--------|-------------------|------------|------------|
| User platziert falsches Modell (nicht CTranslate2) | Mittel | Hoch | Klare Dokumentation + Fehlermeldung im Dialog |
| Kein Auto-Download -> User frustriert | Niedrig | Mittel | Freundlicher Dialog mit detaillierter Anleitung |
| Dateigröße (User muss zusätzliches Modell downloaden) | Garantiert | Niedrig | ZIP bleibt klein (~200 MB), nur einmaliger Download |
| Path-Redirect-Bug in portable Mode | Niedrig | Hoch | Test auf Windows mit portable Build |
| /health Endpunkt incompatible mit Frontend | Niedrig | Mittel | Test der JSON-Struktur |

---

## Offene Punkte (post-Implementierung)

- [ ] Test auf Windows mit portable Build
- [ ] Test mit verschiedenen Modellen (tiny, small, large-v3-turbo)
- [ ] Test der Fehlerdialog-Nutzbarkeit
- [ ] Überprüfung der Dokumentation auf Klarheit
- [ ] Performance-Test mit large-v3-turbo auf CPU/GPU

---

**Plan-Version:** 1.0  
**Datum:** 2026-06-30  
**Status:** Bereit zur Umsetzung ✅