"""
LLM Text Correction (Phase 3D — Lemonade + Gemma 4 NPU).

Post-Processing des transkribierten Textes mit Gemma 4 (E2B/E4B) via Lemonade-
Server (localhost:8000). Lemonade nutzt FastFlowLM auf dem XDNA 2 NPU für
maximalen Durchsatz (~28 tok/s) und niedrige Latenz (~0,5-1,5s Korrekturzeit).

Architektur (Phase 3D, Option B — NPU-only):
  - Dieser Modul ist ein reiner HTTP-Client für Lemonade's OpenAI-kompatible API.
  - Alle Modell-/NPU-/Quantisierungs-Logik liegt in FastFlowLM (Lemonade Sidecar).
  - Kein onnxruntime, kein Qwen3, kein KV-Cache, kein Sampling mehr hier.
  - Bei JEDEM Fehler (Lemonade nicht erreichbar, NPU fehlt, Netzwerkfehler)
    wird der Originaltext unveraendert zurueckgegeben.

Der System-Prompt ist editierbar und wird in der zentralen config.json
persistiert. Das technische Geruest (Rollen-Markup) bleibt fix.

Robustheit: bei JEDEM Fehler wird der Originaltext unveraendert zurueckgegeben —
die Transkriptions-Pipeline stuerzt niemals wegen der Korrektur ab.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

# Lemonade-Server Konfiguration
LEMONADE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "gemma4-it:e2b"
HEALTH_TIMEOUT = 2          # Sekunden für Health-Check
REQUEST_TIMEOUT = 15        # Sekunden für Korrektur-Request
MAX_PROMPT_CHARS = 2000

DEFAULT_SYSTEM_PROMPT = (
    "Korrigiere Rechtschreibung und Grammatik des folgenden Textes. "
    "Entferne Fuellwoerter wie aehm oder hm. "
    "Behalte Sinn und Wortstellung bei. "
    "Antworte nur mit dem korrigierten Text."
)

# ---------------------------------------------------------------------------
# Editierbarer System-Prompt (Inhalt)
# ---------------------------------------------------------------------------
system_prompt: str = DEFAULT_SYSTEM_PROMPT


def set_system_prompt(prompt) -> str:
    """Setzt den System-Prompt (Inhalt). Leer -> Default, Cap MAX_PROMPT_CHARS."""
    global system_prompt
    p = (prompt or "").strip()
    if not p:
        p = DEFAULT_SYSTEM_PROMPT
    if len(p) > MAX_PROMPT_CHARS:
        p = p[:MAX_PROMPT_CHARS]
    system_prompt = p
    return system_prompt


def get_system_prompt() -> str:
    return system_prompt


# ---------------------------------------------------------------------------
# Lemonade HTTP-Client
# ---------------------------------------------------------------------------


class LemonadeCorrector:
    """Reiner HTTP-Client für Lemonade's OpenAI-kompatible API.
    Alle Modell-/NPU-Logik liegt in FastFlowLM (Lemonade Sidecar).
    
    Bei erstem correct() ohne geladenes Modell wird automatisch ein
    Pull-Request an Lemonade gesendet (Modell-Download startet im
    Hintergrund, dauert 1-3 Minuten)."""

    def __init__(self, base_url: str = LEMONADE_URL,
                 model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._available: bool | None = None
        self._last_check: float = 0.0
        self._pull_triggered: bool = False

    def is_available(self) -> bool:
        """Health-Check via GET /v1/models (mit 5s-Cache, um Latenz zu sparen)."""
        now = time.time()
        if self._available is not None and (now - self._last_check) < 5.0:
            return self._available
        try:
            r = requests.get(f"{self.base_url}/models",
                             timeout=HEALTH_TIMEOUT)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        self._last_check = now
        if not self._available:
            logger.debug("🍋 Lemonade nicht verfügbar — LLM-Korrektur deaktiviert")
        return self._available

    def _model_loaded(self) -> bool:
        """Prüft via GET /v1/models ob das aktive Modell heruntergeladen ist."""
        try:
            r = requests.get(f"{self.base_url}/models",
                             timeout=HEALTH_TIMEOUT)
            models = r.json().get("data", [])
            return any(m.get("id") == self.model for m in models)
        except Exception:
            return False

    def _trigger_model_pull(self):
        """Startet POST /v1/pull im Hintergrund (fire-and-forget).
        Wird nur einmal pro Session ausgelöst."""
        if self._pull_triggered:
            return
        self._pull_triggered = True
        logger.info(f"🍋 Modell-Download gestartet: {self.model} (~1,5 GB, dauert 1-3 Minuten)")
        try:
            requests.post(
                f"{self.base_url}/pull",
                json={"model_name": self.model},
                timeout=1,  # Nicht auf Antwort warten
            )
        except Exception:
            pass  # Pull läuft im Hintergrund, Fehler ignorieren

    def correct(self, text: str) -> str:
        """POST /v1/chat/completions mit System-Prompt + User-Text.
        Bei JEDEM Fehler -> Originaltext (Fallback-Ebene 2: keine Korrektur).
        Bei nicht geladenem Modell: Pull starten, Rohtext zurückgeben."""
        if not text or not text.strip():
            return text
        if not self.is_available():
            return text

        # Prüfen ob Modell geladen ist — falls nicht, Pull starten
        if not self._model_loaded():
            self._trigger_model_pull()
            return text

        prompt = (system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
        max_tokens = min(1024, max(64, len(text) * 2))
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text.strip()},
                    ],
                    "temperature": 0.3,
                    "max_tokens": max_tokens,
                },
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"].strip()
            if not out:
                logger.info("✨ Lemonade lieferte leer — Rohtext")
                return text
            logger.info(
                f"✨ Lemonade ({self.model}): "
                f"'{text.strip()[:60]}' -> '{out[:60]}'"
            )
            return out
        except Exception as e:
            logger.warning(f"⚠️ LLM-Korrektur fehlgeschlagen: {e} — liefere Rohtext")
            # Nächster Health-Check erzwingen (evtl. ist Lemonade gecrasht)
            self._available = None
            return text


# Modul-Singleton (lazy, Lemonade muss laufen)
lemonade = LemonadeCorrector()


def correct_text(text: str) -> str:
    """Komfort-Wrapper für die Transkriptions-Pipeline."""
    return lemonade.correct(text)
