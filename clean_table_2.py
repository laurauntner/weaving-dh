"""
Finalize the Weaving DH tables after manual review.
"""

import csv
import sys
from collections import OrderedDict

DEFAULT_FULL_IN = "../FULL Weaving DH Data Table.csv"
DEFAULT_CLEAN_IN = "../CLEAN Weaving DH Data Table.csv"
DEFAULT_FULL_OUT = "../FULL Weaving DH Data Table.csv"
DEFAULT_CLEAN_OUT = "../CLEAN Weaving DH Data Table.csv"

ID_COL = "id"
WORDS_COL = "textile_words"
NOTES_COL = "notes_metaphorical_usage_textile"

IRRELEVANT_CATEGORIES = {
    "tech jargon",
    "textile reference",
    "verb",
    "unrelated",
    "other language",
}


def load_rows(path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        return list(reader.fieldnames), list(reader)


def write_rows(path: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_word_category_map(clean_rows: list[dict[str, str]]) -> dict[str, "OrderedDict[str, str]"]:
    """id -> {word: category}, in order of first appearance in the CLEAN table."""
    per_id: dict[str, OrderedDict[str, str]] = {}
    for row in clean_rows:
        pid, word, category = row[ID_COL], row[WORDS_COL].strip(), row[NOTES_COL]
        bucket = per_id.setdefault(pid, OrderedDict())
        if word not in bucket:
            bucket[word] = category
        elif bucket[word] != category:
            print(f"warning: id={pid} word={word!r} has conflicting categories "
                  f"({bucket[word]!r} vs {category!r}) - keeping the first")
    return per_id


def sync_full_table(full_rows: list[dict[str, str]], per_id: dict[str, "OrderedDict[str, str]"]) -> int:
    full_ids = {row[ID_COL] for row in full_rows}
    updated = 0

    for row in full_rows:
        words_categories = per_id.get(row[ID_COL])
        if words_categories is None:
            continue
        for word, category in words_categories.items():
            if category and category.upper().startswith("XXXX"):
                print(f"warning: id={row[ID_COL]} word={word!r} still has an "
                      f"unresolved marker ({category!r}) - writing through as-is")
        row[WORDS_COL] = ", ".join(words_categories.keys())
        row[NOTES_COL] = ", ".join(words_categories.values())
        updated += 1

    orphan_ids = set(per_id) - full_ids
    for pid in sorted(orphan_ids):
        print(f"warning: id={pid} exists in the CLEAN table but not in the full table - skipped")

    return updated


def filter_irrelevant_categories(clean_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    kept = [r for r in clean_rows if r[NOTES_COL].strip().lower() not in IRRELEVANT_CATEGORIES]
    return kept, len(clean_rows) - len(kept)


def main(argv: list[str]) -> int:
    if len(argv) == 5:
        full_in, clean_in, full_out, clean_out = argv[1:5]
    elif len(argv) == 1:
        full_in, clean_in, full_out, clean_out = (
            DEFAULT_FULL_IN, DEFAULT_CLEAN_IN, DEFAULT_FULL_OUT, DEFAULT_CLEAN_OUT,
        )
    else:
        print("usage: finalize_weaving_tables.py [full_in clean_in full_out clean_out]")
        return 1

    full_fields, full_rows = load_rows(full_in)
    clean_fields, clean_rows = load_rows(clean_in)

    per_id = build_word_category_map(clean_rows)
    updated_ids = sync_full_table(full_rows, per_id)
    write_rows(full_out, full_fields, full_rows)

    kept_rows, removed = filter_irrelevant_categories(clean_rows)
    write_rows(clean_out, clean_fields, kept_rows)

    print(f"full table:  {updated_ids} ids synced -> {full_out}")
    print(f"clean table: {removed} rows removed, {len(kept_rows)} kept -> {clean_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))