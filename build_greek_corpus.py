"""build_greek_corpus.py — assemble input.txt from public-domain Greek tales.

Downloads English retellings and prose translations of Greek mythology and the
Homeric epics from Project Gutenberg, strips the Gutenberg header/footer
boilerplate, and concatenates them into input.txt. Fits the name: Posaidon
(Poseidon) trained on the sea-god's own myths. Run once to (re)create the
(gitignored) training corpus:

    python build_greek_corpus.py
"""

import re
import urllib.request

# Project Gutenberg ebook ids -> title (English, public domain Greek myth/epic)
WORKS = {
    3327: "Bulfinch — The Age of Fable",
    22381: "Berens — Myths and Legends of Greece and Rome",
    677: "Kingsley — The Heroes",
    976: "Hawthorne — Tanglewood Tales",
    2199: "Homer — The Iliad (Butler)",
    1727: "Homer — The Odyssey (Butler)",
    1728: "Homer — The Odyssey (Butcher & Lang)",
    348: "Hesiod, the Homeric Hymns, and Homerica",
}

START = re.compile(r"\*\*\* START OF TH[EI]S? PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
END = re.compile(r"\*\*\* END OF TH[EI]S? PROJECT GUTENBERG EBOOK", re.S)
# leftover credits / transcriber notes that sit inside the START/END body
CREDIT = re.compile(r"Produced by .*?(pgdp\.net[^\n]*(\s*\(This file.*?\))?|\n)", re.S)
NOTE = re.compile(r"\[\s*(Illustration|Transcriber's Note).*?\]", re.S)


def strip_boilerplate(text):
    """Keep only the work between Gutenberg's START/END markers, minus the
    proofreading credit and bracketed illustration/transcription notes."""
    text = START.split(text, 1)[-1]
    text = END.split(text, 1)[0]
    text = CREDIT.sub("", text)
    text = NOTE.sub("", text)
    return text.strip()


def main():
    parts = []
    for book_id, title in WORKS.items():
        url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
        raw = urllib.request.urlopen(url).read().decode("utf-8")
        body = strip_boilerplate(raw)
        print(f"{title:46s} {len(body):>9,d} chars")
        parts.append(body)

    corpus = "\n\n\n".join(parts) + "\n"
    with open("input.txt", "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"\nwrote input.txt: {len(corpus):,d} chars")


if __name__ == "__main__":
    main()
