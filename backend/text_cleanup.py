"""
Heuristic Text Cleanup (Phase 3A).

Entfernt typische Sprech-Fragmente aus ASR-Rohtext:
  1. Fülllaute (ähm, äh, hmm, ...)
  2. Wortwiederholungen ("ich ich gehe" -> "ich gehe", auch "ich, ich gehe")
  3. Stotter-Fragmente ("g-gehe" -> "gehe" — nur wenn das Fragment den
     Wortanfang bildet, damit echte Bindestrich-Wörter wie "E-Mail"
     erhalten bleiben)
  4. Whitespace-/Satzzeichen-Aufräumarbeiten (mit Abkürzungs-Schutz:
     "z.B." bleibt unverändert, "!?" und "..." bleiben erhalten)

Keine ML-Abhängigkeit, keine Modelle, 0 ms Latenz.
"""

import re
import logging

logger = logging.getLogger(__name__)

_FILLER_SOUNDS = {
    "ähm", "äh", "ah", "oh", "uh", "mhm", "hmm", "mmm", "hm",
    "ähs", "ähms", "naja", "tja", "soo", "ohh", "ahh",
    # Phase 3A-Erweiterung: weitere Sprechlaute
    "ähahm", "ähäh", "öh", "öhm", "mhh", "hmpf",
}

_FILLER_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(w) for w in sorted(_FILLER_SOUNDS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)

# 2a: Duplikate mit "weichen" Trennern (Leerzeichen, Komma, Semikolon,
# Doppelpunkt, Gedankenstrich). KEINE Satzende-Zeichen (.!?) — damit bleiben
# legitime Wiederholungen ueber Satzgrenzen hinweg ("Ja. Ja, morgen.") erhalten.
_DUPLICATE_PATTERN = re.compile(
    r'\b(\w+)(?:[\s,;:—–-]+\1\b)+',
    re.IGNORECASE,
)

# 1a: Stotter-Fragment: 1-2 Buchstaben + Bindestrich + Folgewort (Lookahead,
# wird nicht konsumiert — nur das Fragment wird entfernt).
# Das Fragment wird NUR entfernt, wenn es der Anfang des Folgewortes ist
# ("g-gehe" -> "gehe"). Echte Bindestrich-Wörter ("E-Mail", "U-Bahn")
# bleiben dadurch erhalten.
_STUTTER_PATTERN = re.compile(
    r'\b([A-Za-zÄÖÜäöüß]{1,2})-(?=([A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß]*))',
)

# Ellipse temporaer schuetzen (drei ASCII-Punkte), damit 1c sie nicht zerstoert
_ELLIPSIS = "…"
_ELLIPSIS_PATTERN = re.compile(r'\.\.\.')


def clean_text(text: str) -> str:
    """Wendet heuristische Cleanup-Regeln auf Rohtext an."""
    if not text or not text.strip():
        return text

    original = text

    text = _remove_filler_sounds(text)
    text = _remove_stuttering(text)
    # Stotter-Entfernung KANN neue Duplikate erzeugen ("g-gehe" + vorhandenes
    # "gehe" -> "gehe gehe"), deshalb Duplikate NACH dem Stottern.
    text = _remove_duplicates(text)
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
    def _repl(m):
        fragment = m.group(1)
        folgewort = m.group(2)
        # Nur entfernen, wenn das Fragment der Wortanfang des Folgewortes ist
        if folgewort.lower().startswith(fragment.lower()):
            return ''
        return m.group(0)  # echte Bindestrich-Wörter beibehalten

    text = _STUTTER_PATTERN.sub(_repl, text)
    return text


def _polish(text: str) -> str:
    # Ellipse schuetzen, bevor Satzzeichen-Regeln greifen
    text = _ELLIPSIS_PATTERN.sub(_ELLIPSIS, text)

    text = re.sub(r'\s{2,}', ' ', text)
    # Leerzeichen zwischen Satzzeichen entfernen, Satzzeichen BEIDE behalten
    # ("Was ! ?" -> "Was!?")
    text = re.sub(r'([,.!?;:])\s+(?=[,.!?;:])', r'\1', text)
    # 1c: NUR identische Satzzeichen reduzieren ("!!" -> "!", "??" -> "?")
    # Verschiedene Kombinationen bleiben erhalten ("!?" ist legitim).
    # Ellipsen sind durch _ELLIPSIS geschuetzt.
    text = re.sub(r'([,.!?;:])\1+', r'\1', text)
    # Leerzeichen vor Satzzeichen entfernen
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    # 1b: Leerzeichen nach Satzzeichen NUR einfügen, wenn davor ein Wort mit
    # mindestens 3 Buchstaben endet — schützt Abkürzungen ("z.B.", "d.h.")
    text = re.sub(r'(?<=[A-Za-zÄÖÜäöüß]{3})([,.!?;:])(?=[A-Za-zÄÖÜäöüß])', r'\1 ', text)
    # 2b: Führende Rest-Trenner nach Füllwort-Entfernung strippen
    # (", ich gehe" -> "ich gehe"; "…" = geschuetzte Ellipse ebenfalls)
    text = re.sub(r'^[\s,.!?;:—–-…]+', '', text)

    # Ellipse zurueckkonvertieren
    text = text.replace(_ELLIPSIS, '...')

    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def apply_dictionary(text: str, rules) -> str:
    """Wendet User-Wortersetzungen (Wörterbuch) auf den Text an.

    - Case-insensitive
    - Wortgrenzen (\\b): ersetzt keine Teilwörter ("hans" trifft nicht "Hänschen")
    - Sonderzeichen werden regex-escaped ("(c)" matcht literal)
    - Mehrwort-Phrasen funktionieren über die Wortgrenzen
    - Leere 'from'-Regeln und ungültige Einträge werden übersprungen
    """
    if not text:
        return text
    for rule in rules or []:
        try:
            frm = rule.get("from")
            to = rule.get("to")
        except AttributeError:
            continue
        frm = (frm or "").strip()
        if not frm:
            continue
        to = to or ""
        # Wortgrenzen als negative Lookbehind/Lookahead statt \\b: verhindert
        # Teilwort-Treffer ("hans" != "Hanswurst"), funktioniert aber auch fuer
        # Regeln mit Sonderzeichen an Anfang/Ende ("(c)", "v3.2")
        pattern = re.compile(
            r'(?<![A-Za-zÄÖÜäöüß0-9_])' + re.escape(frm) + r'(?![A-Za-zÄÖÜäöüß0-9_])',
            re.IGNORECASE,
        )
        text = pattern.sub(to, text)
    return text
