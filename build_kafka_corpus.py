"""build_kafka_corpus.py — assemble input.txt from public-domain Kafka texts.

Downloads selected German Kafka works from Project Gutenberg, strips the
Gutenberg header/footer boilerplate, and concatenates them into input.txt.
Run once to (re)create the (gitignored) training corpus:

    python build_kafka_corpus.py
"""

import re
import urllib.request

# Project Gutenberg ebook ids -> title (German Kafka, public domain)
WORKS = {
    69327: "Der Prozess",
    22367: "Die Verwandlung",
    21989: "Ein Landarzt",
    21593: "Das Urteil",
    25791: "In der Strafkolonie",
    30655: "Ein Hungerkünstler",
    23532: "Betrachtung",
    16304: "Der Heizer",
}

START = re.compile(r"\*\*\* START OF TH[EI]S? PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
END = re.compile(r"\*\*\* END OF TH[EI]S? PROJECT GUTENBERG EBOOK", re.S)
# leftover credits / transcriber notes that sit inside the START/END body
CREDIT = re.compile(r"Produced by .*?(pgdp\.net[^\n]*(\s*\(This file.*?\))?|\n)", re.S)
NOTE = re.compile(r"\[\s*(Anmerkungen zur Transkription|Transcriber's Note).*?\]", re.S)


def strip_boilerplate(text):
    """Keep only Kafka's text between Gutenberg's START/END markers,
    minus the proofreading credit and bracketed transcription notes."""
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
        print(f"{title:22s} {len(body):>8,d} chars")
        parts.append(body)

    corpus = "\n\n\n".join(parts) + "\n"
    with open("input.txt", "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"\nwrote input.txt: {len(corpus):,d} chars")


if __name__ == "__main__":
    main()
