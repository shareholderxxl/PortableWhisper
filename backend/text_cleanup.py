"""
Heuristic Text Cleanup (Phase 3A).

Entfernt typische Sprech-Fragmente aus ASR-Rohtext:
  1. Fülllaute (ähm, äh, hmm, ...)
  2. Wortwiederholungen ("ich ich gehe" -> "ich gehe")
  3. Stotter-Fragmente ("g-gehe" -> "gehe")
  4. Whitespace-/Satzzeichen-Aufräumarbeiten

Keine ML-Abhängigkeit, keine Modelle, 0 ms Latenz.
"""

import re
import logging

logger = logging.getLogger(__name__)

_FILLER_SOUNDS = {
    "ähm", "äh", "ah", "oh", "uh", "mhm", "hmm", "mmm", "hm",
    "ähs", "ähms", "naja", "tja", "soo", "ohh", "ahh",
}

_FILLER_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(w) for w in sorted(_FILLER_SOUNDS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)

_DUPLICATE_PATTERN = re.compile(r'\b(\w+)(?:\s+\1\b)+', re.IGNORECASE)

_STUTTER_PATTERN = re.compile(r'\b[A-Za-zÄÖÜäöüß]{1,2}-(?=[A-Za-zÄÖÜäöüß])')


def clean_text(text: str) -> str:
    """Wendet heuristische Cleanup-Regeln auf Rohtext an."""
    if not text or not text.strip():
        return text

    original = text

    text = _remove_filler_sounds(text)
    text = _remove_duplicates(text)
    text = _remove_stuttering(text)
    text = _polish(text)

    if text != original:
        logger.info(f"🧹 Cleanup: '{original.strip()[:80]}' -> '{text[:80]}'")

    return text


def _remove_filler_sounds(text: str) -> str:
    text = _FILLER_PATTERN.sub('', text)
    return text


def _remove_duplicates(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = _DUPLICATE_PATTERN.sub(r'\1', text)
    return text


def _remove_stuttering(text: str) -> str:
    text = _STUTTER_PATTERN.sub('', text)
    return text


def _polish(text: str) -> str:
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'([,.!?;:])\s+(?=[,.!?;:])', '', text)
    text = re.sub(r'([,.!?;:])(?=[,.!?;:])', '', text)
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])(?=[A-Za-zÄÖÜäöüß])', r'\1 ', text)
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text
