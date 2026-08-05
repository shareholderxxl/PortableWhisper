"""
Text Cleanup Regression Test (Phase 3A Verbesserungen).

Manuell ausfuehrbar:  python test_text_cleanup.py
(kein pytest — im Stil von test_gpu_capabilities.py)

Deckt ab:
  - Falsch-Positive-Schutz: E-Mail, U-Bahn, z.B., !?, Ellipsen
  - Sprechmuster-Korrektur: ich ich, ich, ich, g-gehe
  - Füllwort-Entfernung inkl. führende Rest-Trenner
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from text_cleanup import clean_text


def check(name: str, inp: str, expected: str) -> bool:
    out = clean_text(inp)
    ok = out == expected
    status = "✅" if ok else "❌"
    logger.info(f"{status} {name}: {inp!r} -> {out!r}  (erwartet: {expected!r})")
    return ok


def run_all() -> bool:
    logger.info("=" * 70)
    logger.info("Text Cleanup Regression Tests")
    logger.info("=" * 70)

    results = []

    # --- Kategorie 1a: Stotter-Regel ---
    results.append(check("Stotter einfach", "g-gehe", "Gehe"))
    results.append(check("Stotter mehrsilbig", "st-statt", "Statt"))
    results.append(check("Stotter grosz geschrieben", "G-gehe", "Gehe"))
    results.append(check("E-Mail bleibt", "E-Mail", "E-Mail"))
    results.append(check("U-Bahn bleibt", "U-Bahn", "U-Bahn"))
    results.append(check("T-Shirt bleibt", "T-Shirt", "T-Shirt"))
    results.append(check("E-Zigarette bleibt", "E-Zigarette", "E-Zigarette"))
    results.append(check("E-Einkauf ist Stotter", "e-Einkauf", "Einkauf"))

    # --- Kategorie 1b: Abkürzungs-Schutz ---
    results.append(check("z.B. bleibt", "das ist z.B. gut", "Das ist z.B. gut"))
    results.append(check("d.h. bleibt", "man sagt d.h. so", "Man sagt d.h. so"))
    results.append(check("U.S.A. bleibt", "in den U.S.A. leben", "In den U.S.A. leben"))
    results.append(check("Satzpunkt + Wort", "Hallo.Wie geht es", "Hallo. Wie geht es"))
    results.append(check("Komma + Wort", "Hallo,wie geht es", "Hallo, wie geht es"))

    # --- Kategorie 1c: Satzzeichen ---
    results.append(check("Ausrufezeichen-Fragezeichen bleibt", "Wirklich!?", "Wirklich!?"))
    results.append(check("Frage-Ausrufe bleibt", "Was?!", "Was?!"))
    results.append(check("Ellipse bleibt", "das ist gut...", "Das ist gut..."))
    results.append(check("Doppeltes Ausrufezeichen", "Das!!", "Das!"))
    results.append(check("Doppeltes Fragezeichen", "Was??", "Was?"))
    results.append(check("Doppeltes Komma", "Hallo,, Welt", "Hallo, Welt"))
    results.append(check("Leerzeichen zwischen Satzzeichen", "Was ! ?", "Was!?"))

    # --- Kategorie 2a: Duplikate ---
    results.append(check("Duplikat einfach", "ich ich gehe", "Ich gehe"))
    results.append(check("Duplikat mit Komma", "ich, ich gehe", "Ich gehe"))
    results.append(check("Duplikat mit Semikolon", "das; das Ding", "Das Ding"))
    results.append(check("Duplikat Kette", "ich, ich, ich gehe", "Ich gehe"))
    results.append(check("Satzgrenze respektiert", "Ja. Ja, morgen komme ich.", "Ja. Ja, morgen komme ich."))
    results.append(check("Kein Duplikat bei verschiedenen Wörtern", "ich du wir", "Ich du wir"))

    # --- Kategorie 2b: Führende Rest-Trenner ---
    results.append(check("Füllwort + Komma am Anfang", "ähm, ich gehe", "Ich gehe"))
    results.append(check("Füllwort + Punkt am Anfang", "äh. Das ist gut.", "Das ist gut."))
    results.append(check("Füllwort + Ellipse am Anfang", "ähm ... ich gehe", "Ich gehe"))

    # --- Kategorie 3: Füllwörter ---
    results.append(check("Klassisches Füllwort Mitte", "das ist ähm gut", "Das ist gut"))
    results.append(check("Neues Füllwort öhm", "öhm ja", "Ja"))
    results.append(check("Neues Füllwort ähahm", "ähahm also", "Also"))
    results.append(check("Neues Füllwort hmpf", "hmpf, na gut", "Na gut"))
    results.append(check("Mehrere Füllwörter", "ähm hmm das", "Das"))
    results.append(check("Also bleibt als Wort", "also das", "Also das"))

    # --- Feinschliff ---
    results.append(check("Großschreibung Satzanfang", "ich gehe nach hause", "Ich gehe nach hause"))
    results.append(check("Leerzeichen bereinigt", "das   ist   gut", "Das ist gut"))
    results.append(check("Komplettes Beispiel", "ähm ich ich gehe g-gehe zum einkauf", "Ich gehe zum einkauf"))

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
