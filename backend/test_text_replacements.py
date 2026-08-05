"""
Wörterbuch (Wortersetzungen) Regression Test (Phase 3E).

Manuell ausfuehrbar:  python test_text_replacements.py
(kein pytest — im Stil von test_gpu_capabilities.py / test_text_cleanup.py)
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from text_cleanup import apply_dictionary


def check(name: str, inp: str, rules, expected: str) -> bool:
    out = apply_dictionary(inp, rules)
    ok = out == expected
    status = "✅" if ok else "❌"
    logger.info(f"{status} {name}: {inp!r} -> {out!r}  (erwartet: {expected!r})")
    return ok


def run_all() -> bool:
    logger.info("=" * 70)
    logger.info("Wörterbuch (Wortersetzungen) Tests")
    logger.info("=" * 70)

    results = []

    # --- Einfache Ersetzung ---
    results.append(check("Einfach", "hans ist da", [{"from": "hans", "to": "Hans"}], "Hans ist da"))
    results.append(check("Case-insensitive", "Hans geht", [{"from": "hans", "to": "Hans"}], "Hans geht"))
    results.append(check("Wortgrenze - kein Teilwort", "Hanswurst ist da", [{"from": "hans", "to": "Hans"}], "Hanswurst ist da"))
    results.append(check("Wortgrenze - Ende", "der hans", [{"from": "hans", "to": "Hans"}], "der Hans"))
    results.append(check("Wortgrenze - Komma", "hans, komm", [{"from": "hans", "to": "Hans"}], "Hans, komm"))

    # --- Sonderzeichen (regex-escape) ---
    results.append(check("Klammern literal", "das ist (c) echt", [{"from": "(c)", "to": "©"}], "das ist © echt"))
    results.append(check("Punkt literal", "v3.2 ist neu", [{"from": "v3.2", "to": "Version 3.2"}], "Version 3.2 ist neu"))

    # --- Mehrwort-Phrasen ---
    results.append(check("Mehrwort-Phrase", "portable whisper ist toll", [{"from": "portable whisper", "to": "PortableWhisper"}], "PortableWhisper ist toll"))
    results.append(check("Phrase case-insensitive", "Portable Whisper läuft", [{"from": "portable whisper", "to": "PortableWhisper"}], "PortableWhisper läuft"))

    # --- Mehrere Regeln nacheinander ---
    results.append(check("Zwei Regeln", "hans trifft fritz", [
        {"from": "hans", "to": "Hans"},
        {"from": "fritz", "to": "Fritz"},
    ], "Hans trifft Fritz"))

    # --- Ungültige Regeln ---
    results.append(check("Leere from-Regel uebersprungen", "hans bleibt", [{"from": "  ", "to": "X"}], "hans bleibt"))
    results.append(check("Kein dict-Eintrag uebersprungen", "hans bleibt", ["kaputt"], "hans bleibt"))
    results.append(check("Rules=None", "hans bleibt", None, "hans bleibt"))
    results.append(check("Rules=[]", "hans bleibt", [], "hans bleibt"))

    # --- Leerer Text ---
    results.append(check("Leerer Text", "", [{"from": "hans", "to": "Hans"}], ""))
    results.append(check("Whitespace-Text", "   ", [{"from": "hans", "to": "Hans"}], "   "))

    logger.info("=" * 70)
    failed = [r for r in results if not r]
    if failed:
        logger.error(f"❌ {len(failed)}/{len(results)} Tests fehlgeschlagen")
        return False
    logger.info(f"✅ Alle {len(results)} Tests bestanden")
    return True


if __name__ == "__main__":
    ok = run_all()
    exit(0 if ok else 1)
