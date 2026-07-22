use std::sync::Arc;
use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Manager, AppHandle, State,
};
use windows::Win32::{
    UI::Input::KeyboardAndMouse::{
        SendInput, INPUT, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP,
        VK_CONTROL, VK_V, KEYEVENTF_EXTENDEDKEY,
    },
    System::DataExchange::{
        OpenClipboard, CloseClipboard, EmptyClipboard, SetClipboardData, GetClipboardData,
    },
    System::Memory::{GlobalAlloc, GlobalLock, GlobalUnlock, GlobalSize, GMEM_MOVEABLE},
    System::Registry::{RegOpenKeyExW, RegSetValueExW, RegDeleteValueW, RegQueryValueExW, RegCloseKey, HKEY_CURRENT_USER, KEY_READ, KEY_WRITE, REG_SZ},
    Foundation::{HWND, HANDLE, HGLOBAL},
};
use tokio::sync::Mutex;
use anyhow::Result;
use tauri_plugin_global_shortcut::{Code, Modifiers, Shortcut, GlobalShortcutExt};
use std::path::PathBuf;
use std::fs;

// Persistent settings structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Settings {
    pub selected_model: String,
    pub selected_device: String,
    pub selected_microphone: Option<i32>,
    pub use_clipboard: bool,
    pub selected_language: String,
    pub toggle_shortcut: String,
    pub cancel_shortcut: String,
    pub recording_mode: String,
    pub lemonade_enabled: bool,       // Phase 3D: LLM-Korrektur via Lemonade/NPU
    pub lemonade_model: String,       // Phase 3D: "gemma4-it:e2b" | "gemma4-it:e4b"
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            selected_model: "default".to_string(),
            selected_device: "auto".to_string(),
            selected_microphone: None,
            use_clipboard: true,
            selected_language: "de".to_string(),
            toggle_shortcut: "F9".to_string(),
            cancel_shortcut: "Escape".to_string(),
            recording_mode: "toggle".to_string(),
            lemonade_enabled: true,
            lemonade_model: "gemma4-it:e2b".to_string(),
        }
    }
}

// Simple state - track model, device, and clipboard setting
#[derive(Debug, Clone)]
pub struct AppState {
    pub selected_model: Arc<Mutex<String>>,
    pub selected_device: Arc<Mutex<String>>,
    pub selected_microphone: Arc<Mutex<Option<i32>>>,  // Microphone device index (None = default)
    pub use_clipboard: Arc<Mutex<bool>>,  // New: whether to paste to clipboard
    pub selected_language: Arc<Mutex<String>>,  // Selected language code
    pub toggle_shortcut: Arc<Mutex<String>>,  // Toggle recording shortcut
    pub cancel_shortcut: Arc<Mutex<String>>,  // Cancel recording shortcut
    pub backend_child: Arc<Mutex<Option<tauri_plugin_shell::process::CommandChild>>>,  // Backend process handle
    pub lemonade_child: Arc<Mutex<Option<tauri_plugin_shell::process::CommandChild>>>,  // Lemonade process handle (Phase 3D)
    pub is_processing: Arc<Mutex<bool>>,  // Track if we are transcribing
    pub recording_mode: Arc<Mutex<String>>,  // Toggle vs Push-to-Talk
    pub settings_path: PathBuf,  // Path to settings file
    pub lemonade_enabled: Arc<Mutex<bool>>,  // Phase 3D: Lemonade/NPU LLM correction on/off
    pub lemonade_model: Arc<Mutex<String>>,  // Phase 3D: Active Gemma 4 model id
}

impl AppState {
    fn new(settings_path: PathBuf) -> Self {
        Self {
            selected_model: Arc::new(Mutex::new("default".to_string())),
            selected_device: Arc::new(Mutex::new("auto".to_string())),
            selected_microphone: Arc::new(Mutex::new(None)),
            use_clipboard: Arc::new(Mutex::new(true)),
            selected_language: Arc::new(Mutex::new("de".to_string())),
            toggle_shortcut: Arc::new(Mutex::new("F9".to_string())),
            cancel_shortcut: Arc::new(Mutex::new("Escape".to_string())),
            backend_child: Arc::new(Mutex::new(None)),
            lemonade_child: Arc::new(Mutex::new(None)),
            is_processing: Arc::new(Mutex::new(false)),
            recording_mode: Arc::new(Mutex::new("toggle".to_string())),
            settings_path,
            lemonade_enabled: Arc::new(Mutex::new(true)),
            lemonade_model: Arc::new(Mutex::new("gemma4-it:e2b".to_string())),
        }
    }

    async fn load_settings(&self) -> Settings {
        match fs::read_to_string(&self.settings_path) {
            Ok(content) => {
                match serde_json::from_str::<Settings>(&content) {
                    Ok(settings) => {
                        log::info!("✅ Loaded settings from {:?}", self.settings_path);
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

    async fn save_settings(&self) {
        let settings = Settings {
            selected_model: self.selected_model.lock().await.clone(),
            selected_device: self.selected_device.lock().await.clone(),
            selected_microphone: *self.selected_microphone.lock().await,
            use_clipboard: *self.use_clipboard.lock().await,
            selected_language: self.selected_language.lock().await.clone(),
            toggle_shortcut: self.toggle_shortcut.lock().await.clone(),
            cancel_shortcut: self.cancel_shortcut.lock().await.clone(),
            recording_mode: self.recording_mode.lock().await.clone(),
            lemonade_enabled: *self.lemonade_enabled.lock().await,
            lemonade_model: self.lemonade_model.lock().await.clone(),
        };

        if let Ok(json) = serde_json::to_string_pretty(&settings) {
            if let Err(e) = fs::write(&self.settings_path, json) {
                log::error!("❌ Failed to save settings: {}", e);
            } else {
                log::info!("💾 Settings saved to {:?}", self.settings_path);
            }
        }
    }

    async fn apply_settings(&self, settings: Settings) {
        *self.selected_model.lock().await = settings.selected_model;
        *self.selected_device.lock().await = settings.selected_device;
        *self.selected_microphone.lock().await = settings.selected_microphone;
        *self.use_clipboard.lock().await = settings.use_clipboard;
        *self.selected_language.lock().await = settings.selected_language;
        *self.toggle_shortcut.lock().await = settings.toggle_shortcut;
        *self.cancel_shortcut.lock().await = settings.cancel_shortcut;
        *self.recording_mode.lock().await = settings.recording_mode;
        *self.lemonade_enabled.lock().await = settings.lemonade_enabled;
        *self.lemonade_model.lock().await = settings.lemonade_model;
    }
}

// Get current clipboard content (UTF-16 text)
fn get_clipboard_text() -> Option<Vec<u16>> {
    unsafe {
        const CF_UNICODETEXT: u32 = 13;
        
        if let Err(_) = OpenClipboard(HWND::default()) {
            return None;
        }

        let h_clipboard_data = match GetClipboardData(CF_UNICODETEXT) {
            Ok(handle) if !handle.is_invalid() => handle,
            _ => {
                let _ = CloseClipboard();
                return None;
            }
        };

        // Convert HANDLE to HGLOBAL
        let hglobal = HGLOBAL(h_clipboard_data.0 as _);
        
        let locked = GlobalLock(hglobal);
        if locked.is_null() {
            let _ = CloseClipboard();
            return None;
        }

        let size = GlobalSize(hglobal);
        if size == 0 {
            let _ = GlobalUnlock(hglobal);
            let _ = CloseClipboard();
            return None;
        }
        
        let mut data = vec![0u16; size / 2];
        std::ptr::copy_nonoverlapping(locked as *const u16, data.as_mut_ptr(), size / 2);

        let _ = GlobalUnlock(hglobal);
        let _ = CloseClipboard();

        Some(data)
    }
}

// Set clipboard text (UTF-16)
fn set_clipboard_text(text_utf16: &[u16]) -> Result<()> {
    unsafe {
        if let Err(e) = OpenClipboard(HWND::default()) {
            return Err(anyhow::anyhow!("Failed to open clipboard: {}", e));
        }

        if let Err(e) = EmptyClipboard() {
            let _ = CloseClipboard();
            return Err(anyhow::anyhow!("Failed to empty clipboard: {}", e));
        }

        let len = text_utf16.len() * std::mem::size_of::<u16>();
        let hmem = GlobalAlloc(GMEM_MOVEABLE, len)
            .map_err(|e| anyhow::anyhow!("Failed to allocate memory: {}", e))?;

        let locked = GlobalLock(hmem);
        if locked.is_null() {
            let _ = CloseClipboard();
            return Err(anyhow::anyhow!("Failed to lock memory"));
        }

        std::ptr::copy_nonoverlapping(text_utf16.as_ptr(), locked as *mut u16, text_utf16.len());
        let _ = GlobalUnlock(hmem);

        const CF_UNICODETEXT: u32 = 13;
        let result = SetClipboardData(CF_UNICODETEXT, HANDLE(hmem.0 as _));
        if let Err(e) = result {
            let _ = CloseClipboard();
            return Err(anyhow::anyhow!("Failed to set clipboard data: {}", e));
        }

        let _ = CloseClipboard();
        Ok(())
    }
}

// Text injection via clipboard with optional clipboard preservation
pub fn inject_text(text: &str, save_to_clipboard: bool) -> Result<()> {
    unsafe {
        // Save old clipboard content if we need to restore it
        let old_clipboard = if !save_to_clipboard {
            get_clipboard_text()
        } else {
            None
        };

        // Prepare text as UTF-16
        let mut text_utf16: Vec<u16> = text.encode_utf16().collect();
        text_utf16.push(0);

        // Set clipboard with new text
        set_clipboard_text(&text_utf16)?;

        // Wait for clipboard to update
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Simulate Ctrl+V
        let inputs = vec![
            INPUT {
                r#type: INPUT_KEYBOARD,
                Anonymous: windows::Win32::UI::Input::KeyboardAndMouse::INPUT_0 {
                    ki: KEYBDINPUT { wVk: VK_CONTROL, wScan: 0, dwFlags: KEYEVENTF_EXTENDEDKEY, time: 0, dwExtraInfo: 0 },
                },
            },
            INPUT {
                r#type: INPUT_KEYBOARD,
                Anonymous: windows::Win32::UI::Input::KeyboardAndMouse::INPUT_0 {
                    ki: KEYBDINPUT { wVk: VK_V, wScan: 0, dwFlags: KEYEVENTF_EXTENDEDKEY, time: 0, dwExtraInfo: 0 },
                },
            },
            INPUT {
                r#type: INPUT_KEYBOARD,
                Anonymous: windows::Win32::UI::Input::KeyboardAndMouse::INPUT_0 {
                    ki: KEYBDINPUT { wVk: VK_V, wScan: 0, dwFlags: KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, time: 0, dwExtraInfo: 0 },
                },
            },
            INPUT {
                r#type: INPUT_KEYBOARD,
                Anonymous: windows::Win32::UI::Input::KeyboardAndMouse::INPUT_0 {
                    ki: KEYBDINPUT { wVk: VK_CONTROL, wScan: 0, dwFlags: KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, time: 0, dwExtraInfo: 0 },
                },
            },
        ];

        SendInput(&inputs, std::mem::size_of::<INPUT>() as i32);

        // Restore old clipboard if needed
        if !save_to_clipboard {
            if let Some(old_text) = old_clipboard {
                // Wait a bit for paste to complete
                std::thread::sleep(std::time::Duration::from_millis(50));
                let _ = set_clipboard_text(&old_text);
                log::info!("📋 Clipboard restored to previous content");
            } else {
                // If there was no previous clipboard content, clear it
                std::thread::sleep(std::time::Duration::from_millis(50));
                let empty: Vec<u16> = vec![0];
                let _ = set_clipboard_text(&empty);
                log::info!("📋 Clipboard cleared");
            }
        } else {
            log::info!("📋 Text saved to clipboard and pasted");
        }
    }

    Ok(())
}

// Simple command: Inject text (always injects, optionally saves to clipboard)
#[tauri::command]
async fn inject_text_directly(text: String, save_to_clipboard: bool) -> Result<(), String> {
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    inject_text(&text, save_to_clipboard).map_err(|e| e.to_string())?;
    log::info!("✅ Injected: {} (clipboard: {})", text, if save_to_clipboard { "saved" } else { "not saved" });
    Ok(())
}

// Simple command: Start recording
#[tauri::command]
async fn cmd_start_recording(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    log::info!("═══════════════════════════════════════════════");
    log::info!("🎬 START RECORDING");
    log::info!("═══════════════════════════════════════════════");

    let model = state.selected_model.lock().await.clone();
    let device = state.selected_device.lock().await.clone();
    let microphone = state.selected_microphone.lock().await.clone();
    let language = state.selected_language.lock().await.clone();

    // Position window at top center and show
    if let Some(win) = app.get_webview_window("recording") {
        // Get primary monitor to calculate center position
        if let Some(monitor) = win.current_monitor().map_err(|e| e.to_string())? {
            let screen_size = monitor.size();
            let window_size = win.outer_size().map_err(|e| e.to_string())?;

            // Calculate top-right position (20px from edges)
            let x = (screen_size.width as i32 - window_size.width as i32) - 20;
            let y = 20;

            win.set_position(tauri::PhysicalPosition::new(x, y)).map_err(|e| e.to_string())?;
        }

        win.show().map_err(|e| e.to_string())?;

        // Play start sound
        let _ = win.eval("playStartSound()");

        log::info!("✅ Window shown at top center");
    }

    // Call backend /start
    let client = reqwest::Client::new();
    tokio::spawn(async move {
        // Use None for auto-detect, otherwise use the selected language
        let lang_value = if language == "auto" {
            serde_json::Value::Null
        } else {
            serde_json::json!(language)
        };

        let mut request_body = serde_json::json!({
            "model_size": model,
            "language": lang_value,
            "device": device
        });

        // Add device_index if a specific microphone is selected
        if let Some(device_index) = microphone {
            request_body["device_index"] = serde_json::json!(device_index);
        }

        match client.post("http://127.0.0.1:8765/start")
            .json(&request_body)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => log::info!("✅ Backend started"),
            Ok(resp) => log::error!("❌ Backend error: {}", resp.status()),
            Err(e) => log::error!("❌ Request failed: {}", e),
        }
    });

    Ok(())
}

// Simple command: Cancel recording
#[tauri::command]
async fn cmd_cancel_recording(app: AppHandle) -> Result<(), String> {
    log::info!("═══════════════════════════════════════════════");
    log::info!("❌ CANCEL RECORDING");
    log::info!("═══════════════════════════════════════════════");

    // Call backend /cancel
    let client = reqwest::Client::new();
    tokio::spawn(async move {
        match client.post("http://127.0.0.1:8765/cancel")
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => log::info!("✅ Backend cancelled"),
            Ok(resp) => log::error!("❌ Backend error: {}", resp.status()),
            Err(e) => log::error!("❌ Request failed: {}", e),
        }
    });

    // Hide window
    if let Some(win) = app.get_webview_window("recording") {
        win.hide().map_err(|e| e.to_string())?;
        log::info!("✅ Window hidden");
    }

    Ok(())
}

// Simple command: Stop recording (called by F9 when window visible)
#[tauri::command]
async fn cmd_stop_recording(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    log::info!("═══════════════════════════════════════════════");
    log::info!("🛑 STOP RECORDING");
    log::info!("═══════════════════════════════════════════════");

    // Set processing state to true
    {
        let mut is_proc = state.is_processing.lock().await;
        *is_proc = true;
    }

    // Hide window FIRST (to restore focus to text field immediately!)
    if let Some(win) = app.get_webview_window("recording") {
        let _ = win.eval("showProcessing()");
        let _ = win.eval("playStopSound()");
        log::info!("📢 Called showProcessing() in frontend");
        
        if let Err(e) = win.hide() {
            log::error!("❌ Failed to hide recording window: {}", e);
            {
                let mut is_proc = state.is_processing.lock().await;
                *is_proc = false;
            }
            return Err(e.to_string());
        }
        log::info!("✅ Window hidden immediately");
    }

    // Small delay to let frontend update UI
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Call backend /stop to get transcription
    let client = reqwest::Client::new();
    let text_to_inject = match client.post("http://127.0.0.1:8765/stop")
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            log::info!("✅ Backend stopped");

            // Get transcription text
            if let Ok(data) = resp.json::<serde_json::Value>().await {
                if let Some(text) = data.get("text").and_then(|t| t.as_str()) {
                    log::info!("📝 Transcription: {}", text);
                    Some(text.to_string())
                } else {
                    None
                }
            } else {
                None
            }
        }
        Ok(resp) => {
            log::error!("❌ Backend error: {}", resp.status());
            None
        }
        Err(e) => {
            log::error!("❌ Request failed: {}", e);
            None
        }
    };

    // Wait for focus to return to the text field (if not already returned)
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;

    // THEN inject text (always inject, clipboard setting controls how to inject)
    if let Some(text) = text_to_inject {
        let save_to_clipboard = *state.use_clipboard.lock().await;
        log::info!("🔧 Clipboard save setting: {}", save_to_clipboard);
        
        if let Err(e) = inject_text(&text, save_to_clipboard) {
            log::error!("❌ Injection failed: {}", e);
        } else {
            log::info!("✅ Text injected (clipboard: {})", if save_to_clipboard { "saved" } else { "restored" });
        }
    }

    // Reset processing state to false
    {
        let mut is_proc = state.is_processing.lock().await;
        *is_proc = false;
    }

    Ok(())
}

// F9 shortcut handler
#[tauri::command]
async fn cmd_toggle_recording(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    log::info!("⌨️ F9 PRESSED");

    // Check if we are currently transcribing/processing
    {
        let is_processing = state.is_processing.lock().await;
        if *is_processing {
            log::info!("   Already processing/transcribing, ignoring toggle request");
            return Ok(());
        }
    }

    if let Some(win) = app.get_webview_window("recording") {
        let is_visible = win.is_visible().unwrap_or(false);
        log::info!("   Window visible: {}", is_visible);

        if is_visible {
            // Stop - call backend /stop, transcribe, and inject
            cmd_stop_recording(app, state).await?;
        } else {
            // Start
            cmd_start_recording(app, state).await?;
        }
    }

    Ok(())
}

// Get current model and device settings
#[tauri::command]
async fn get_model_and_device(state: State<'_, AppState>) -> Result<(String, String), String> {
    let model = state.selected_model.lock().await.clone();
    let device = state.selected_device.lock().await.clone();
    Ok((model, device))
}

// Set microphone device
#[tauri::command]
async fn set_microphone_device(
    device_index: Option<i32>,
    state: State<'_, AppState>
) -> Result<(), String> {
    *state.selected_microphone.lock().await = device_index;
    log::info!("🎤 Microphone device set to: {:?}", device_index);
    state.save_settings().await;
    Ok(())
}

// Get microphone device
#[tauri::command]
async fn get_microphone_device(state: State<'_, AppState>) -> Result<Option<i32>, String> {
    Ok(*state.selected_microphone.lock().await)
}

// New: Set clipboard paste setting
#[tauri::command]
async fn set_clipboard_paste(
    enabled: bool,
    state: State<'_, AppState>
) -> Result<(), String> {
    *state.use_clipboard.lock().await = enabled;
    log::info!("⚙️ Clipboard paste setting: {}", enabled);
    state.save_settings().await;
    Ok(())
}

// New: Get clipboard paste setting
#[tauri::command]
async fn get_clipboard_paste(state: State<'_, AppState>) -> Result<bool, String> {
    let enabled = *state.use_clipboard.lock().await;
    Ok(enabled)
}

// Language commands
#[tauri::command]
async fn set_language(language: String, state: State<'_, AppState>) -> Result<(), String> {
    *state.selected_language.lock().await = language.clone();
    log::info!("🌐 Language set to: {}", language);
    state.save_settings().await;
    Ok(())
}

#[tauri::command]
async fn get_language(state: State<'_, AppState>) -> Result<String, String> {
    Ok(state.selected_language.lock().await.clone())
}

// Recording mode commands
#[tauri::command]
async fn get_recording_mode(state: State<'_, AppState>) -> Result<String, String> {
    Ok(state.recording_mode.lock().await.clone())
}

#[tauri::command]
async fn set_recording_mode(mode: String, state: State<'_, AppState>) -> Result<(), String> {
    *state.recording_mode.lock().await = mode.clone();
    log::info!("⚙️ Recording mode set to: {}", mode);
    state.save_settings().await;
    Ok(())
}

// Open URL in default browser (Windows command helper)
#[tauri::command]
async fn open_url(url: String) -> Result<(), String> {
    log::info!("🔗 Opening URL: {}", url);
    if url.starts_with("http://") || url.starts_with("https://") {
        let _ = std::process::Command::new("cmd")
            .args(&["/C", "start", "", &url])
            .spawn();
    }
    Ok(())
}

// Helper function to parse shortcut string to Shortcut object
fn parse_shortcut(shortcut_str: &str) -> Option<Shortcut> {

    let parts: Vec<&str> = shortcut_str.split('+').collect();
    let mut modifiers = Modifiers::empty();
    let mut key_code: Option<Code> = None;

    for part in parts {
        let part = part.trim();
        match part {
            "Ctrl" | "Control" => modifiers |= Modifiers::CONTROL,
            "Alt" => modifiers |= Modifiers::ALT,
            "Shift" => modifiers |= Modifiers::SHIFT,
            "Super" | "Win" | "Meta" => modifiers |= Modifiers::SUPER,
            // Function keys
            "F1" => key_code = Some(Code::F1),
            "F2" => key_code = Some(Code::F2),
            "F3" => key_code = Some(Code::F3),
            "F4" => key_code = Some(Code::F4),
            "F5" => key_code = Some(Code::F5),
            "F6" => key_code = Some(Code::F6),
            "F7" => key_code = Some(Code::F7),
            "F8" => key_code = Some(Code::F8),
            "F9" => key_code = Some(Code::F9),
            "F10" => key_code = Some(Code::F10),
            "F11" => key_code = Some(Code::F11),
            "F12" => key_code = Some(Code::F12),
            // Special keys
            "Escape" | "Esc" => key_code = Some(Code::Escape),
            "Space" => key_code = Some(Code::Space),
            "Tab" => key_code = Some(Code::Tab),
            "Enter" | "Return" => key_code = Some(Code::Enter),
            "Backspace" => key_code = Some(Code::Backspace),
            "Delete" => key_code = Some(Code::Delete),
            "Insert" => key_code = Some(Code::Insert),
            "Home" => key_code = Some(Code::Home),
            "End" => key_code = Some(Code::End),
            "PageUp" => key_code = Some(Code::PageUp),
            "PageDown" => key_code = Some(Code::PageDown),
            // Arrow keys
            "ArrowUp" | "Up" => key_code = Some(Code::ArrowUp),
            "ArrowDown" | "Down" => key_code = Some(Code::ArrowDown),
            "ArrowLeft" | "Left" => key_code = Some(Code::ArrowLeft),
            "ArrowRight" | "Right" => key_code = Some(Code::ArrowRight),
            // Letter keys (single character)
            s if s.len() == 1 && s.chars().next().unwrap().is_alphabetic() => {
                let ch = s.chars().next().unwrap().to_ascii_uppercase();
                key_code = match ch {
                    'A' => Some(Code::KeyA),
                    'B' => Some(Code::KeyB),
                    'C' => Some(Code::KeyC),
                    'D' => Some(Code::KeyD),
                    'E' => Some(Code::KeyE),
                    'F' => Some(Code::KeyF),
                    'G' => Some(Code::KeyG),
                    'H' => Some(Code::KeyH),
                    'I' => Some(Code::KeyI),
                    'J' => Some(Code::KeyJ),
                    'K' => Some(Code::KeyK),
                    'L' => Some(Code::KeyL),
                    'M' => Some(Code::KeyM),
                    'N' => Some(Code::KeyN),
                    'O' => Some(Code::KeyO),
                    'P' => Some(Code::KeyP),
                    'Q' => Some(Code::KeyQ),
                    'R' => Some(Code::KeyR),
                    'S' => Some(Code::KeyS),
                    'T' => Some(Code::KeyT),
                    'U' => Some(Code::KeyU),
                    'V' => Some(Code::KeyV),
                    'W' => Some(Code::KeyW),
                    'X' => Some(Code::KeyX),
                    'Y' => Some(Code::KeyY),
                    'Z' => Some(Code::KeyZ),
                    _ => None,
                };
            }
            // Number keys
            s if s.len() == 1 && s.chars().next().unwrap().is_numeric() => {
                let ch = s.chars().next().unwrap();
                key_code = match ch {
                    '0' => Some(Code::Digit0),
                    '1' => Some(Code::Digit1),
                    '2' => Some(Code::Digit2),
                    '3' => Some(Code::Digit3),
                    '4' => Some(Code::Digit4),
                    '5' => Some(Code::Digit5),
                    '6' => Some(Code::Digit6),
                    '7' => Some(Code::Digit7),
                    '8' => Some(Code::Digit8),
                    '9' => Some(Code::Digit9),
                    _ => None,
                };
            }
            // Special symbol keys
            "\\" | "Backslash" => key_code = Some(Code::Backslash),
            "/" | "Slash" => key_code = Some(Code::Slash),
            ";" | "Semicolon" => key_code = Some(Code::Semicolon),
            "'" | "Quote" => key_code = Some(Code::Quote),
            "[" | "BracketLeft" => key_code = Some(Code::BracketLeft),
            "]" | "BracketRight" => key_code = Some(Code::BracketRight),
            "," | "Comma" => key_code = Some(Code::Comma),
            "." | "Period" => key_code = Some(Code::Period),
            "`" | "Backquote" => key_code = Some(Code::Backquote),
            "-" | "Minus" => key_code = Some(Code::Minus),
            "=" | "Equal" => key_code = Some(Code::Equal),
            _ => {
                log::warn!("⚠️ Unknown key: {}", part);
            }
        }
    }

    if let Some(code) = key_code {
        Some(Shortcut::new(Some(modifiers), code))
    } else {
        None
    }
}

// Shortcut commands
#[tauri::command]
async fn save_shortcuts(
    shortcuts: std::collections::HashMap<String, String>,
    app: AppHandle,
    state: State<'_, AppState>
) -> Result<(), String> {
    let mut settings_changed = false;

    // Handle toggle shortcut
    if let Some(toggle) = shortcuts.get("toggle") {
        let old_shortcut = state.toggle_shortcut.lock().await.clone();
        *state.toggle_shortcut.lock().await = toggle.clone();
        log::info!("⌨️ Toggle shortcut saved: {} (was: {})", toggle, old_shortcut);
        settings_changed = true;

        // Re-register the shortcut
        // First, unregister old shortcut
        if let Some(old_sc) = parse_shortcut(&old_shortcut) {
            if let Err(e) = app.global_shortcut().unregister(old_sc) {
                log::warn!("⚠️ Failed to unregister old toggle shortcut {}: {}", old_shortcut, e);
            } else {
                log::info!("✅ Unregistered old toggle shortcut: {}", old_shortcut);
            }
        }

        // Register new shortcut
        if let Some(new_sc) = parse_shortcut(toggle) {
            if let Err(e) = app.global_shortcut().register(new_sc) {
                log::error!("❌ Failed to register new toggle shortcut {}: {}", toggle, e);
                return Err(format!("Failed to register toggle shortcut: {}", e));
            } else {
                log::info!("✅ Registered new toggle shortcut: {}", toggle);
            }
        } else {
            log::error!("❌ Failed to parse toggle shortcut: {}", toggle);
            return Err(format!("Invalid toggle shortcut format: {}", toggle));
        }
    }

    // Handle cancel shortcut
    if let Some(cancel) = shortcuts.get("cancel") {
        let old_shortcut = state.cancel_shortcut.lock().await.clone();
        *state.cancel_shortcut.lock().await = cancel.clone();
        log::info!("⌨️ Cancel shortcut saved: {} (was: {})", cancel, old_shortcut);
        settings_changed = true;

        // Re-register the shortcut
        // First, unregister old shortcut
        if let Some(old_sc) = parse_shortcut(&old_shortcut) {
            if let Err(e) = app.global_shortcut().unregister(old_sc) {
                log::warn!("⚠️ Failed to unregister old cancel shortcut {}: {}", old_shortcut, e);
            } else {
                log::info!("✅ Unregistered old cancel shortcut: {}", old_shortcut);
            }
        }

        // Register new shortcut
        if let Some(new_sc) = parse_shortcut(cancel) {
            if let Err(e) = app.global_shortcut().register(new_sc) {
                log::error!("❌ Failed to register new cancel shortcut {}: {}", cancel, e);
                return Err(format!("Failed to register cancel shortcut: {}", e));
            } else {
                log::info!("✅ Registered new cancel shortcut: {}", cancel);
            }
        } else {
            log::error!("❌ Failed to parse cancel shortcut: {}", cancel);
            return Err(format!("Invalid cancel shortcut format: {}", cancel));
        }
    }

    // Save settings if changed
    if settings_changed {
        state.save_settings().await;
    }

    Ok(())
}

#[tauri::command]
async fn get_toggle_shortcut(state: State<'_, AppState>) -> Result<String, String> {
    Ok(state.toggle_shortcut.lock().await.clone())
}

#[tauri::command]
async fn get_cancel_shortcut(state: State<'_, AppState>) -> Result<String, String> {
    Ok(state.cancel_shortcut.lock().await.clone())
}

// Stub commands for settings that don't need backend implementation yet
#[tauri::command]
async fn get_preferred_languages() -> Result<Vec<String>, String> {
    Ok(vec![])  // Not used anymore, but kept for compatibility
}

#[tauri::command]
async fn set_preferred_languages(_languages: Vec<String>) -> Result<(), String> {
    Ok(())  // Not used anymore, but kept for compatibility
}

#[tauri::command]
async fn get_launch_on_startup() -> Result<bool, String> {
    use windows::core::PCWSTR;

    unsafe {
        const REG_PATH: &str = "Software\\Microsoft\\Windows\\CurrentVersion\\Run";
        const APP_NAME: &str = "PortableWhisper";

        // Convert registry path to wide string
        let reg_path_wide: Vec<u16> = REG_PATH.encode_utf16().chain(std::iter::once(0)).collect();

        // Open registry key
        let mut hkey = std::mem::zeroed();
        let result = RegOpenKeyExW(
            HKEY_CURRENT_USER,
            PCWSTR::from_raw(reg_path_wide.as_ptr()),
            0,
            KEY_READ,
            &mut hkey,
        );

        if result.is_err() {
            log::info!("🔍 Launch on startup: Not set (registry key not accessible)");
            return Ok(false);
        }

        // Query for our app name
        let app_name_wide: Vec<u16> = APP_NAME.encode_utf16().chain(std::iter::once(0)).collect();
        let mut data_size: u32 = 0;

        let query_result = RegQueryValueExW(
            hkey,
            PCWSTR::from_raw(app_name_wide.as_ptr()),
            None,
            None,
            None,
            Some(&mut data_size),
        );

        let _ = RegCloseKey(hkey);

        let exists = query_result.is_ok() && data_size > 0;
        log::info!("🔍 Launch on startup: {}", if exists { "Enabled" } else { "Disabled" });
        Ok(exists)
    }
}

#[tauri::command]
async fn set_launch_on_startup(enabled: bool) -> Result<(), String> {
    use windows::core::PCWSTR;

    unsafe {
        const REG_PATH: &str = "Software\\Microsoft\\Windows\\CurrentVersion\\Run";
        const APP_NAME: &str = "PortableWhisper";

        // Convert registry path to wide string
        let reg_path_wide: Vec<u16> = REG_PATH.encode_utf16().chain(std::iter::once(0)).collect();

        // Open registry key with write access
        let mut hkey = std::mem::zeroed();
        let result = RegOpenKeyExW(
            HKEY_CURRENT_USER,
            PCWSTR::from_raw(reg_path_wide.as_ptr()),
            0,
            KEY_WRITE,
            &mut hkey,
        );

        if result.is_err() {
            return Err("Failed to open registry key".to_string());
        }

        let app_name_wide: Vec<u16> = APP_NAME.encode_utf16().chain(std::iter::once(0)).collect();

        if enabled {
            // Get current executable path
            let exe_path = std::env::current_exe()
                .map_err(|e| format!("Failed to get executable path: {}", e))?;

            let exe_path_str = format!("\"{}\"", exe_path.display());
            let exe_path_wide: Vec<u16> = exe_path_str.encode_utf16().chain(std::iter::once(0)).collect();

            // Set registry value
            let set_result = RegSetValueExW(
                hkey,
                PCWSTR::from_raw(app_name_wide.as_ptr()),
                0,
                REG_SZ,
                Some(&std::slice::from_raw_parts(
                    exe_path_wide.as_ptr() as *const u8,
                    exe_path_wide.len() * 2,
                )),
            );

            let _ = RegCloseKey(hkey);

            if set_result.is_ok() {
                log::info!("✅ Launch on startup: Enabled ({})", exe_path_str);
                Ok(())
            } else {
                Err(format!("Failed to set registry value: {:?}", set_result))
            }
        } else {
            // Delete registry value (ignore error if value doesn't exist)
            let _ = RegDeleteValueW(
                hkey,
                PCWSTR::from_raw(app_name_wide.as_ptr()),
            );

            let _ = RegCloseKey(hkey);

            // Success if deletion worked or if value didn't exist
            log::info!("✅ Launch on startup: Disabled");
            Ok(())
        }
    }
}

// Restart backend command
#[tauri::command]
async fn restart_backend(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    log::info!("🔄 Restarting backend...");

    // Kill existing backend if running
    if let Some(child) = state.backend_child.lock().await.take() {
        log::info!("🛑 Killing existing backend process...");
        match child.kill() {
            Ok(_) => {
                log::info!("✅ Backend process killed");
                // Give it a moment to fully terminate
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            }
            Err(e) => {
                log::warn!("⚠️ Failed to kill backend: {}", e);
            }
        }
    }

    // Start new backend
    log::info!("🚀 Starting new backend process...");
    use tauri_plugin_shell::ShellExt;

    let sidecar_command = app.shell()
        .sidecar("binaries/whisper-backend")
        .map_err(|e| format!("Failed to create sidecar command: {}", e))?;

    let (_rx, child) = sidecar_command
        .spawn()
        .map_err(|e| format!("Failed to spawn backend sidecar: {}", e))?;

    // Store the new child process
    *state.backend_child.lock().await = Some(child);

    // Wait for backend to start
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    log::info!("✅ Backend restarted successfully");

    Ok(())
}

// ===========================================================================
// Phase 3D: Lemonade Sidecar (LLM Korrektur via NPU + Gemma 4)
// ===========================================================================

/// Health-Check: ist Lemonade auf Port 8000 erreichbar?
async fn check_lemonade_health() -> bool {
    match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
    {
        Ok(client) => match client.get("http://127.0.0.1:8000/v1/models").send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        },
        Err(_) => false,
    }
}

/// Startet den Lemonade-Sidecar, falls nicht bereits aktiv und falls aktiviert.
/// Gibt true zurück, wenn Lemonade läuft (oder schon lief).
async fn ensure_lemonade_running(app: &AppHandle, state: &AppState) -> bool {
    // Prüfen, ob Lemonade deaktiviert ist
    if !*state.lemonade_enabled.lock().await {
        log::info!("🍋 Lemonade deaktiviert (config) — LLM-Korrektur aus");
        return false;
    }
    // Bereits aktiv?
    if check_lemonade_health().await {
        return true;
    }
    // Bestehenden Child-Handle aufräumen (falls der Prozess zuvor gecrasht ist)
    if state.lemonade_child.lock().await.is_some() {
        *state.lemonade_child.lock().await = None;
    }
    // Sidecar spawnen
    use tauri_plugin_shell::ShellExt;
    let sidecar = app.shell().sidecar("binaries/lemond");
    let sidecar = match sidecar {
        Ok(cmd) => cmd,
        Err(e) => {
            log::warn!("⚠️ Lemonade-Binary nicht gefunden: {}", e);
            return false;
        }
    };
    let (rx, child) = match sidecar.spawn() {
        Ok(pair) => pair,
        Err(e) => {
            log::warn!("⚠️ Lemonade-Sidecar konnte nicht gestartet werden: {}", e);
            return false;
        }
    };
    *state.lemonade_child.lock().await = Some(child);
    // Output-Events loggen (optional, wie bei backend)
    let _ = rx;
    log::info!("🍋 Lemonade-Sidecar gestartet, warte auf Health-Check...");
    // Health-Check mit Timeout (bis zu 15s)
    for attempt in 1..=15 {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        if check_lemonade_health().await {
            log::info!("✅ Lemonade ist nach {}s bereit", attempt);
            return true;
        }
    }
    log::warn!("⚠️ Lemonade nicht innerhalb 15s bereit — LLM-Korrektur deaktiviert");
    false
}

#[tauri::command]
async fn get_llm_provider_status(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let enabled = *state.lemonade_enabled.lock().await;
    let model = state.lemonade_model.lock().await.clone();
    let healthy = if enabled { check_lemonade_health().await } else { false };
    Ok(serde_json::json!({
        "provider": if healthy { "lemonade" } else { "none" },
        "model": model,
        "enabled": enabled,
        "health": healthy,
    }))
}

#[tauri::command]
async fn set_lemonade_model(model: String, state: State<'_, AppState>) -> Result<(), String> {
    log::info!("🍋 Setze Lemonade-Modell: {}", model);
    *state.lemonade_model.lock().await = model;
    state.save_settings().await;
    Ok(())
}

#[tauri::command]
async fn set_lemonade_enabled(enabled: bool, state: State<'_, AppState>) -> Result<(), String> {
    log::info!("🍋 Lemonade {}", if enabled { "aktiviert" } else { "deaktiviert" });
    *state.lemonade_enabled.lock().await = enabled;
    state.save_settings().await;
    Ok(())
}

#[tauri::command]
async fn restart_lemonade(app: AppHandle, state: State<'_, AppState>) -> Result<bool, String> {
    log::info!("🔄 Restarting Lemonade...");
    if let Some(child) = state.lemonade_child.lock().await.take() {
        let _ = child.kill();
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }
    let running = ensure_lemonade_running(&app, &state).await;
    Ok(running)
}

// Tray menu
fn create_tray_menu(app: &AppHandle) -> Result<Menu<tauri::Wry>, tauri::Error> {
    let toggle = MenuItem::with_id(app, "toggle", "🎙️ Start/Stop Recording (F9)", true, None::<&str>)?;
    let settings = MenuItem::with_id(app, "settings", "⚙️ Settings", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "❌ Quit", true, None::<&str>)?;
    Menu::with_items(app, &[&toggle, &settings, &quit])
}

fn handle_tray_event(app: &AppHandle, event: TrayIconEvent) {
    // Handle both Up and Down states to be more reliable
    // Use Down for immediate feedback
    match event {
        TrayIconEvent::Click {
            button: MouseButton::Left,
            button_state: MouseButtonState::Down,
            ..
        } => {
            if let Some(win) = app.get_webview_window("main") {
                let _ = if win.is_visible().unwrap_or(false) {
                    win.hide()
                } else {
                    win.show().and_then(|_| win.set_focus())
                };
            }
        }
        _ => {}
    }
}

fn handle_menu_event(app: &AppHandle, event: tauri::menu::MenuEvent) {
    log::info!("📋 Menu clicked: {}", event.id.as_ref());

    match event.id.as_ref() {
        "toggle" => {
            let app_clone = app.clone();
            tauri::async_runtime::spawn(async move {
                let _ = cmd_toggle_recording(app_clone.clone(), app_clone.state()).await;
            });
        }
        "settings" => {
            if let Some(win) = app.get_webview_window("main") {
                let _ = win.show().and_then(|_| win.set_focus());
            }
        }
        "quit" => {
            log::info!("👋 Quit requested via tray menu");
            app.exit(0);
        }
        _ => {}
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            log::info!("🔒 Single instance check - app already running, focusing existing window");
            // Bring main window to front if already running
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .setup(|app| {
            use tauri::WebviewWindowBuilder;

            // Logging
            app.handle().plugin(
                tauri_plugin_log::Builder::default()
                    .level(log::LevelFilter::Info)
                    .target(tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::Stdout))
                    .target(tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::LogDir { file_name: Some("app".to_string()) }))
                    .build(),
            )?;

            log::info!("🚀 PortableWhisper starting...");

            // Initialize app state with settings path in local "data" directory next to the executable
            let exe_dir = std::env::current_exe()
                .expect("Failed to get executable path")
                .parent()
                .expect("Failed to get executable directory")
                .to_path_buf();
            let app_data_dir = exe_dir.join("data");

            // Create local data directory if it doesn't exist
            if !app_data_dir.exists() {
                fs::create_dir_all(&app_data_dir)
                    .expect("Failed to create local data directory");
            }

            let settings_path = app_data_dir.join("settings.json");
            log::info!("📁 Settings path: {:?}", settings_path);

            let app_state = AppState::new(settings_path);

            // Load and apply saved settings
            let settings = tauri::async_runtime::block_on(async {
                app_state.load_settings().await
            });

            tauri::async_runtime::block_on(async {
                app_state.apply_settings(settings.clone()).await;
            });

            log::info!("📋 Loaded settings: model={}, device={}, language={}",
                settings.selected_model, settings.selected_device, settings.selected_language);

            // Store state in app
            app.manage(app_state);

            // Start backend sidecar
            log::info!("🔧 Starting backend server...");
            use tauri_plugin_shell::ShellExt;

            let sidecar_command = app.app_handle()
                .shell()
                .sidecar("binaries/whisper-backend")
                .expect("Failed to create sidecar command");

            let (_rx, child) = sidecar_command
                .spawn()
                .expect("Failed to spawn backend sidecar");

            // Store the child process in state so we can kill it on app exit
            let state: tauri::State<AppState> = app.state();
            tauri::async_runtime::block_on(async {
                *state.backend_child.lock().await = Some(child);
            });

            // Wait a moment for backend to start
            std::thread::sleep(std::time::Duration::from_secs(1));
            log::info!("✅ Backend server started");

            // Phase 3D: Lemonade-Sidecar starten (optional, NPU LLM-Korrektur)
            let lemonade_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let state: tauri::State<AppState> = lemonade_handle.state();
                let _ = ensure_lemonade_running(&lemonade_handle, &state).await;
            });

            // Smart Pre-Load: Start async model load on app startup
            log::info!("📥 Smart Pre-Load: Starting async model load on startup...");
            // Arcs klonen (sind 'static) statt tauri::State (borrow von app)
            let state_preload = app.state::<AppState>();
            let selected_model = state_preload.selected_model.clone();
            let selected_device = state_preload.selected_device.clone();
            let selected_language = state_preload.selected_language.clone();
            tauri::async_runtime::spawn(async move {
                // Get settings from AppState
                let model_size = selected_model.lock().await.clone();
                let device = selected_device.lock().await.clone();
                let language = selected_language.lock().await.clone();

                let lang_value = if language == "auto" {
                    serde_json::Value::Null
                } else {
                    serde_json::json!(language)
                };

                let request_body = serde_json::json!({
                    "model_size": model_size,
                    "language": lang_value,
                    "device": device
                });

                let client = reqwest::Client::new();
                match client.post("http://127.0.0.1:8765/load_model_async")
                    .json(&request_body)
                    .send()
                    .await
                {
                    Ok(resp) if resp.status().is_success() => {
                        log::info!("✅ Smart Pre-Load started successfully (model: {}, device: {})",
                            model_size, device);
                    }
                    Ok(resp) => {
                        log::warn!("⚠️ Smart Pre-Load request failed: {}", resp.status());
                    }
                    Err(e) => {
                        log::warn!("⚠️ Smart Pre-Load request error: {}", e);
                    }
                }
            });

            // Create recording window
            WebviewWindowBuilder::new(app, "recording", tauri::WebviewUrl::App("recording.html".into()))
                .title("Recording")
                .inner_size(300.0, 120.0)
                .resizable(false)
                .position(0.0, 50.0)  // Will be centered horizontally when shown
                .always_on_top(true)
                .visible(false)
                .skip_taskbar(true)
                .decorations(false)
                .transparent(true)
                .focused(false)
                .build()?;

            log::info!("✅ Recording window created");

            // Tray
            let menu = create_tray_menu(app.handle())?;
            let tray = TrayIconBuilder::new()
                .menu(&menu)
                .icon(app.default_window_icon().unwrap().clone())
                .on_menu_event(|app, event| handle_menu_event(app, event))
                .build(app)?;

            let app_handle = app.handle().clone();
            tray.on_tray_icon_event(move |_tray, event| handle_tray_event(&app_handle, event));

            log::info!("✅ Tray icon created");

            // Intercept main window close event to hide instead of destroy
            if let Some(main_window) = app.get_webview_window("main") {
                let app_handle_close = app.handle().clone();
                main_window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        // Prevent window from closing
                        api.prevent_close();
                        // Hide it instead
                        if let Some(win) = app_handle_close.get_webview_window("main") {
                            let _ = win.hide();
                        }
                    }
                });
            }

            // Global shortcuts handler
            let app_handle_hotkey = app.handle().clone();

            app.handle().plugin(
                tauri_plugin_global_shortcut::Builder::new()
                    .with_handler(move |_app, shortcut, event| {
                        use tauri_plugin_global_shortcut::ShortcutState;
                        let is_pressed = event.state == ShortcutState::Pressed;
                        let is_released = event.state == ShortcutState::Released;

                        let app_clone = app_handle_hotkey.clone();
                        let shortcut_str = format!("{:?}", shortcut); // Format outside async block

                        tauri::async_runtime::spawn(async move {
                            let state: tauri::State<AppState> = app_clone.state();
                            let toggle_sc = state.toggle_shortcut.lock().await.clone();
                            let cancel_sc = state.cancel_shortcut.lock().await.clone();
                            let rec_mode = state.recording_mode.lock().await.clone();

                            // Check if this is the cancel shortcut
                            if is_pressed {
                                if let Some(parsed_cancel) = parse_shortcut(&cancel_sc) {
                                    let cancel_str = format!("{:?}", parsed_cancel);
                                    if shortcut_str == cancel_str {
                                        log::info!("🔥 CANCEL SHORTCUT TRIGGERED ({})", cancel_sc);
                                        // Only cancel if recording window is visible
                                        if let Some(win) = app_clone.get_webview_window("recording") {
                                            if win.is_visible().unwrap_or(false) {
                                                let _ = cmd_cancel_recording(app_clone.clone()).await;
                                                return;
                                            }
                                        }
                                    }
                                }
                            }

                            // Check if this is the toggle shortcut
                            if let Some(parsed_toggle) = parse_shortcut(&toggle_sc) {
                                let toggle_str = format!("{:?}", parsed_toggle);
                                if shortcut_str == toggle_str {
                                    if rec_mode == "ptt" {
                                        if is_pressed {
                                            log::info!("🔥 PTT PRESS TRIGGERED ({})", toggle_sc);
                                            let is_proc = state.is_processing.lock().await;
                                            if !*is_proc {
                                                if let Some(win) = app_clone.get_webview_window("recording") {
                                                    let is_visible = win.is_visible().unwrap_or(false);
                                                    if !is_visible {
                                                        let _ = cmd_start_recording(app_clone.clone(), app_clone.state()).await;
                                                    }
                                                }
                                            }
                                        } else if is_released {
                                            log::info!("🔥 PTT RELEASE TRIGGERED ({})", toggle_sc);
                                            if let Some(win) = app_clone.get_webview_window("recording") {
                                                let is_visible = win.is_visible().unwrap_or(false);
                                                if is_visible {
                                                    let _ = cmd_stop_recording(app_clone.clone(), app_clone.state()).await;
                                                }
                                            }
                                        }
                                    } else { // toggle mode
                                        if is_pressed {
                                            log::info!("🔥 TOGGLE SHORTCUT TRIGGERED ({})", toggle_sc);
                                            let _ = cmd_toggle_recording(app_clone.clone(), app_clone.state()).await;
                                        }
                                    }
                                }
                            }
                        });
                    })
                    .build()
            )?;

            // Register initial shortcuts
            let state: tauri::State<AppState> = app.state();
            let (initial_toggle, initial_cancel) = tauri::async_runtime::block_on(async {
                (
                    state.toggle_shortcut.lock().await.clone(),
                    state.cancel_shortcut.lock().await.clone()
                )
            });

            // Register toggle shortcut
            if let Some(toggle_sc) = parse_shortcut(&initial_toggle) {
                if let Err(e) = app.global_shortcut().register(toggle_sc) {
                    log::error!("❌ Failed to register toggle shortcut {}: {}", initial_toggle, e);
                } else {
                    log::info!("✅ Toggle shortcut registered: {}", initial_toggle);
                }
            } else {
                log::error!("❌ Failed to parse initial toggle shortcut: {}", initial_toggle);
            }

            // Register cancel shortcut
            if let Some(cancel_sc) = parse_shortcut(&initial_cancel) {
                if let Err(e) = app.global_shortcut().register(cancel_sc) {
                    log::error!("❌ Failed to register cancel shortcut {}: {}", initial_cancel, e);
                } else {
                    log::info!("✅ Cancel shortcut registered: {}", initial_cancel);
                }
            } else {
                log::error!("❌ Failed to parse initial cancel shortcut: {}", initial_cancel);
            }

            // Notify backend to load the initial model after a short delay
            let app_handle_model = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                // Wait for backend to fully initialize
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

                let state: tauri::State<AppState> = app_handle_model.state();
                let model = state.selected_model.lock().await.clone();
                let device = state.selected_device.lock().await.clone();
                let language = state.selected_language.lock().await.clone();

                let client = reqwest::Client::new();
                let lang_value = if language == "auto" {
                    serde_json::Value::Null
                } else {
                    serde_json::json!(language)
                };

                match client.post("http://127.0.0.1:8765/load_model")
                    .json(&serde_json::json!({
                        "model_size": model,
                        "device": device,
                        "language": lang_value
                    }))
                    .send()
                    .await
                {
                    Ok(resp) if resp.status().is_success() => {
                        log::info!("✅ Initial model loaded: {} on {}", model, device);
                    }
                    Ok(resp) => {
                        log::warn!("⚠️ Backend /load_model returned: {}", resp.status());
                    }
                    Err(e) => {
                        log::warn!("⚠️ Failed to load initial model: {}", e);
                    }
                }
            });

            log::info!("💡 Press F9 to start/stop recording");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            open_url,
            get_recording_mode,
            set_recording_mode,
            inject_text_directly,
            cmd_start_recording,
            cmd_stop_recording,
            cmd_cancel_recording,
            cmd_toggle_recording,
            get_model_and_device,
            set_microphone_device,
            get_microphone_device,
            set_clipboard_paste,
            get_clipboard_paste,
            set_language,
            get_language,
            save_shortcuts,
            get_toggle_shortcut,
            get_cancel_shortcut,
            get_preferred_languages,
            set_preferred_languages,
            get_launch_on_startup,
            set_launch_on_startup,
            restart_backend,
            get_llm_provider_status,
            set_lemonade_model,
            set_lemonade_enabled,
            restart_lemonade
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    // Globaler Exit-Handler: Backend- UND Lemonade-Sidecar bei JEDEM Beendigungspfad sauber
    // killen (Tray-Quit, OS-Shutdown, taskkill, Hauptfenster-Zerstörung),
    // nicht nur im Tray-Quit-Pfad. Verhindert herrenlose whisper-backend.exe / lemond.exe.
    app.run(|app_handle, event| {
        if let tauri::RunEvent::Exit = event {
            let state: tauri::State<AppState> = app_handle.state();
            tauri::async_runtime::block_on(async {
                if let Some(child) = state.backend_child.lock().await.take() {
                    log::info!("🛑 App exiting: killing backend sidecar");
                    if let Err(e) = child.kill() {
                        log::warn!("⚠️ Failed to kill backend on exit: {}", e);
                    } else {
                        log::info!("✅ Backend sidecar killed on exit");
                    }
                }
                if let Some(child) = state.lemonade_child.lock().await.take() {
                    log::info!("🛑 App exiting: killing Lemonade sidecar");
                    if let Err(e) = child.kill() {
                        log::warn!("⚠️ Failed to kill Lemonade on exit: {}", e);
                    } else {
                        log::info!("✅ Lemonade sidecar killed on exit");
                    }
                }
            });
        }
    });
}

