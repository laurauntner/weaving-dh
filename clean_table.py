#!/usr/bin/env python3
"""
Clean and normalize the Weaving DH data table.
"""

import csv
import re
import sys

DEFAULT_INPUT = "../Weaving DH Data Table.csv"
DEFAULT_OUTPUT = "../Weaving DH Data Table clean.csv"

INCLUDE_COL = "include_exclude (default: y)"
WORDS_COL = "textile_words"
KWIC_COL = "kwic_textile"
NOTES_COL = "notes_metaphorical_usage_textile"
NEEDS_REVIEW_MARKER = "XXXX"
KWIC_NEEDS_REVIEW_MARKER = "YYYY"

DROP_COLS = [
    INCLUDE_COL,
    "construction_words",
    "kwic_construction",
    "notes_metaphorical_usage_construction",
]

KWIC_HEADER_RE = re.compile(r"(?im)^\s*([a-zA-Z'-]+)\s*:\s*$")
QUOTE_SEPARATOR_RE = re.compile(r"\n\s*\n\s*\n+")


def split_values(raw: str | None) -> list[str]:
    if raw is None or raw.strip() == "":
        return [raw]
    values = [v.strip() for v in raw.split(",") if v.strip()]
    return values or [raw]


def split_quotes(text: str) -> list[str]:
    if not text:
        return [text]
    quotes = [q.strip() for q in QUOTE_SEPARATOR_RE.split(text) if q.strip()]
    return quotes or [text]


def split_kwic_by_word(kwic: str, words: list[str]) -> dict[str, str]:
    """Map each word to its own kwic content, with the 'word:' header stripped.

    Words without a matching header are absent from the result; callers
    fall back to the full, unsplit kwic text for those.
    """
    word_set = {w.lower() for w in words if w}
    headers = [m for m in KWIC_HEADER_RE.finditer(kwic) if m.group(1).lower() in word_set]

    blocks: dict[str, str] = {}
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(kwic)
        word = m.group(1).lower()
        segment = kwic[m.end():end].strip()
        blocks[word] = f"{blocks[word]}\n\n{segment}" if word in blocks else segment
    return blocks


def determine_note(idx: int, words: list[str], notes: list[str | None], candidates: list[str]) -> str | None:
    if len(notes) == 1:
        note = notes[0]
        return note.lower() if note else note
    if len(notes) == len(words):
        note = notes[idx]
        return note.lower() if note else note
    if candidates:
        return f"{NEEDS_REVIEW_MARKER} {', '.join(candidates)}"
    return NEEDS_REVIEW_MARKER


def build_row(row: dict[str, str], word: str, kwic: str, note: str) -> dict[str, str]:
    new_row = dict(row)
    new_row[WORDS_COL] = word
    new_row[KWIC_COL] = kwic
    new_row[NOTES_COL] = note
    for col in DROP_COLS:
        new_row.pop(col, None)
    return new_row


def expand_row(row: dict[str, str]) -> list[dict[str, str]]:
    words = split_values(row.get(WORDS_COL))
    notes = split_values(row.get(NOTES_COL))
    kwic_full = row.get(KWIC_COL) or ""
    kwic_by_word = split_kwic_by_word(kwic_full, words)
    candidates = [n.lower() for n in notes if n]

    expanded = []
    for idx, word in enumerate(words):
        if len(words) == 1:
            kwic_text, kwic_needs_review = kwic_full, False
        else:
            matched = kwic_by_word.get(word.lower())
            kwic_text, kwic_needs_review = (matched, False) if matched is not None else (kwic_full, True)

        note = determine_note(idx, words, notes, candidates)

        for quote in split_quotes(kwic_text):
            kwic = f"{KWIC_NEEDS_REVIEW_MARKER} {quote}" if kwic_needs_review else quote
            expanded.append(build_row(row, word, kwic, note))
    return expanded


def clean_csv(input_path: str, output_path: str) -> None:
    with open(input_path, newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} has no header row")

        fieldnames = [c for c in reader.fieldnames if c not in DROP_COLS]

        rows_in = 0
        out_rows = []
        for row in reader:
            rows_in += 1
            if (row.get(INCLUDE_COL) or "").strip().lower() != "y":
                continue
            out_rows.extend(expand_row(row))

    with open(output_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"rows read:    {rows_in}")
    print(f"rows written: {len(out_rows)}")
    print(f"output:       {output_path}")


def main(argv: list[str]) -> int:
    if len(argv) == 3:
        input_path, output_path = argv[1], argv[2]
    elif len(argv) == 1:
        input_path, output_path = DEFAULT_INPUT, DEFAULT_OUTPUT
    else:
        print("usage: clean_weaving_csv.py [input.csv output.csv]")
        return 1

    clean_csv(input_path, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))