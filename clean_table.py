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

KWIC_HEADER_RE = re.compile(r"(?im)^\s*[a-zA-Z'-]+\s*:\s*$")
QUOTE_SEPARATOR_RE = re.compile(r"\n\s*\n\s*\n+")
BOLD_TERM_RE = re.compile(r"\*\*([A-Za-z'-]+)\*\*")


def split_values(raw: str | None) -> list[str]:
    if raw is None or raw.strip() == "":
        return [raw]
    values = [v.strip() for v in raw.split(",") if v.strip()]
    return values or [raw]


def word_stems(word: str) -> set[str]:
    """Cheap suffix stripping so 'stitching'/'stitches' match 'stitch', etc."""
    word = word.lower()
    stems = {word}
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[: -len(suffix)]
            stems.add(stem)
            if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                stems.add(stem[:-1])  # doubled consonant, e.g. spinn -> spin
    if word.endswith("ies") and len(word) - 3 >= 2:
        stems.add(word[:-3] + "y")  # tapestries -> tapestry
    return stems


def match_word(bold_term: str, words: list[str]) -> str | None:
    bold_stems = word_stems(bold_term)
    for word in words:
        if bold_stems & word_stems(word):
            return word
        if len(word) >= 4 and bold_term.lower().startswith(word.lower()):
            return word
    return None


def split_kwic_chunks(kwic: str) -> list[str]:
    """Split into individual quotes, stripping any leading 'word:' header line."""
    chunks = [KWIC_HEADER_RE.sub("", part).strip() for part in QUOTE_SEPARATOR_RE.split(kwic)]
    chunks = [c for c in chunks if c]
    return chunks or [kwic]


def assign_quotes_to_words(kwic: str, words: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Attribute each quote to a word via its bolded term.

    Returns (quotes_by_word, unassigned). A word with no matched quote is
    simply absent from quotes_by_word; callers fall back to the full kwic
    text for it. Quotes without a resolvable bold term land in unassigned.
    """
    chunks = split_kwic_chunks(kwic)
    quotes_by_word: dict[str, list[str]] = {}
    unassigned: list[str] = []

    if len(words) == 1:
        return {words[0].lower(): chunks}, []

    for chunk in chunks:
        bold_match = BOLD_TERM_RE.search(chunk)
        matched_word = match_word(bold_match.group(1), words) if bold_match else None
        if matched_word:
            quotes_by_word.setdefault(matched_word.lower(), []).append(chunk)
        else:
            unassigned.append(chunk)
    return quotes_by_word, unassigned


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
    candidates = [n.lower() for n in notes if n]

    quotes_by_word, unassigned = assign_quotes_to_words(kwic_full, words)

    expanded = []
    for idx, word in enumerate(words):
        note = determine_note(idx, words, notes, candidates)
        word_quotes = quotes_by_word.get(word.lower())

        if word_quotes:
            for quote in word_quotes:
                expanded.append(build_row(row, word, quote, note))
        else:
            # no quote could be attributed to this word -> flag for review
            expanded.append(build_row(row, word, f"{KWIC_NEEDS_REVIEW_MARKER} {kwic_full}", note))

        for quote in unassigned:
            expanded.append(build_row(row, word, f"{KWIC_NEEDS_REVIEW_MARKER} {quote}", note))
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