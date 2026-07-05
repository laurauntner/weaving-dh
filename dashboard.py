"""
Build a self-contained, interactive HTML dashboard on metaphorical textile
(and construction) vocabulary in a corpus of digital-humanities journal articles.

Inputs (CSV, see Configuration below):
  * FULL table  — one row per text (article). Holds every annotation: the textile
    and construction words found, their KWIC snippets, the per-word usage
    categories, and the include/exclude decision. This is the source for all
    counts except textile co-occurrence/collocation.
  * CLEAN table — one row per textile-word occurrence, pre-filtered to the
    included texts. Used only for the textile co-occurrence and collocation
    statistics, where per-occurrence granularity matters.

Output: a single HTML file with the computed statistics embedded as JSON and
rendered client-side with Chart.js.

Counting conventions:
  * Rows flagged as a "doublette" in further_notes are dropped before anything
    else, so duplicates never enter any statistic.
  * include_exclude == "y" defines the analysed corpus.
  * Textile usage categories are counted per category per text: within one text
    a category is counted once, however often its words recur. The two metaphor
    categories are restricted to included texts; the seven-category and group
    views additionally read the excluded rows, because the non-metaphor
    categories live almost entirely there.
  * "Metaphorical" vs "Non-metaphorical" is a per-text partition; the headline
    metaphor-text count equals the Metaphorical group exactly.
  * Construction is a single presence flag — a text counts once if it contains a
    construction word, with no sub-categories.
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from typing import Optional
from pathlib import Path

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

for _res, _kind in [("punkt","tokenizers"),("punkt_tab","tokenizers"),
                     ("stopwords","corpora"),("wordnet","corpora"),("omw-1.4","corpora")]:
    try:
        nltk.data.find(f"{_kind}/{_res}")
    except LookupError:
        nltk.download(_res, quiet=True)

# ---------------------------------------------------------------------------
# 0.  Configuration
# ---------------------------------------------------------------------------

CSV_PATH       = Path("../FULL Weaving DH Data Table.csv")
# Clean, one-row-per-occurrence textile KWIC table (already filtered to
# include_exclude == "y"; each row = one textile word hit with its own
# concordance snippet). Used exclusively for the textile co-occurrence
# (Kookurrenz) and collocation / concordance (Konkordanz) statistics — every
# other statistic in the dashboard is computed from CSV_PATH above.
CLEAN_CSV_PATH = Path("../CLEAN Weaving DH Data Table.csv")
OUTPUT_PATH = Path("Weaving DH Dashboard.html")

TOP_N_COOC   = 5
TOP_N_COLLOC = 8
COLLOC_WIN   = 2

STOPWORDS = set(stopwords.words("english")) | {
    "also", "would", "could", "one", "two", "three", "may", "use",
    "used", "using", "well", "within", "across", "however", "thus",
    "therefore", "whether", "though", "even", "much", "many", "first",
    "second", "new", "based", "see", "et", "al", "pp", "fig",
}

# Canonical forms for the textile metaphor usage categories (case-insensitive)
TEXTILE_USAGE_CATEGORIES  = {"textile metaphor", "general metaphor"}

# Canonical display labels (title-case) for each normalised key
TEXTILE_USAGE_LABELS: dict[str, str] = {
    "textile metaphor": "Textile Metaphor",
    "general metaphor": "General Metaphor",
    "textile reference": "Textile Reference",
    "verb":              "Verb",
    "tech jargon":       "Tech Jargon",
    "unrelated":         "Unrelated",
    "other language":    "Other Language",
}

# Top-level classification: which usage category belongs to which group.
# "Metaphorical" are the focus categories used throughout the dashboard;
# "Non-metaphorical" covers everything else — jargon, literal/verb uses,
# and unrelated or foreign-language mentions.
TEXTILE_USAGE_GROUPS: dict[str, str] = {
    "textile metaphor":  "Metaphorical",
    "general metaphor":  "Metaphorical",
    "tech jargon":       "Non-metaphorical",
    "textile reference": "Non-metaphorical",
    "verb":              "Non-metaphorical",
    "unrelated":         "Non-metaphorical",
    "other language":    "Non-metaphorical",
}
TEXTILE_USAGE_GROUP_LABELS = ["Metaphorical", "Non-metaphorical"]
# Full set of textile usage categories (all groups combined). TEXTILE_USAGE_CATEGORIES
# above stays restricted to the two metaphor categories for the per-word and
# per-source breakdowns; this wider set feeds the all-category and group-level views.
ALL_TEXTILE_USAGE_CATEGORIES = set(TEXTILE_USAGE_GROUPS.keys())
CONST_USAGE_LABELS: dict[str, str] = {
    "construction": "Construction",
}

# Surface-form variants that normalise to a canonical category key.
# Covers abbreviations, typos, and alternate phrasings found in the CSV.
TEXTILE_USAGE_VARIANTS: dict[str, str] = {
    "textile metaphor": "textile metaphor",
    "general metaphor": "general metaphor",
    "textile reference": "textile reference",  # kept for display, not a focus category
    "verb":             "verb",
    "tech jargon":      "tech jargon",
    "unrelated":        "unrelated",
    "other language":   "other language",
}

TEXTILE_CANONICAL: dict[str, tuple[str, list[str]]] = {
    "weave":    ("weave",    ["weaving"]),
    "weaving":  ("weave",    ["weaving"]),
    "knit":     ("knit",     ["knitting"]),
    "knitting": ("knit",     ["knitting"]),
    "spin":     ("spin",     ["spinning"]),
    "spinning": ("spin",     ["spinning"]),
    "sew":      ("sew",      ["sewing"]),
    "sewing":   ("sew",      ["sewing"]),
    "stitch":   ("stitch",   ["stitching"]),
    "stitching":("stitch",   ["stitching"]),
    "loom":     ("loom",     []),
    "warp":     ("warp",     []),
    "weft":     ("weft",     []),
    "tapestry": ("tapestry", []),
    "yarn":     ("yarn",     []),
    "thread":   ("thread",   []),
    "fabric":   ("fabric",   []),
    "spindle":  ("spindle",  []),
}

TEXTILE_VARIANTS: dict[str, str] = {
    canon: ", ".join(variants)
    for canon, variants in {
        v[0]: v[1] for v in TEXTILE_CANONICAL.values()
    }.items()
    if variants
}

def textile_canonical(word: str) -> str:
    entry = TEXTILE_CANONICAL.get(word.lower())
    return entry[0] if entry else word.lower()

def parse_usage_categories(cell: str, variant_map: dict[str, str],
                           known_categories: set[str]) -> set[str]:
    """
    Parse a potentially comma-separated category cell into a set of canonical
    category keys.  Each token is looked up in variant_map (case-insensitive);
    tokens not found in the map are silently ignored.  Only keys that are also
    members of known_categories are returned, so callers can restrict to their
    focus set without further filtering.
    """
    if not cell or not cell.strip():
        return set()
    result: set[str] = set()
    for token in cell.split(","):
        normalised = token.strip().lower()
        canonical  = variant_map.get(normalised)
        if canonical and canonical in known_categories:
            result.add(canonical)
    return result

def parse_usage_sequence(cell: str, variant_map: dict[str, str]) -> list[Optional[str]]:
    if not cell or not cell.strip():
        return []
    sequence: list[Optional[str]] = []
    for token in cell.split(","):
        sequence.append(variant_map.get(token.strip().lower()))
    return sequence

_lemmatizer = WordNetLemmatizer()

def lemmatize(word: str) -> str:
    return _lemmatizer.lemmatize(word.lower(), pos="n")

_WORD_RE  = re.compile(r"\b[a-zA-Z]{3,}\b")
_HIT_RE   = re.compile(r"\*\*(\w+)\*\*")
_LABEL_RE = re.compile(r"^\w[\w\s]*:$")

# ---------------------------------------------------------------------------
# 1.  Load and filter CSV
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))

def include_exclude_key(row: dict) -> str:
    for k in row:
        if "include_exclude" in k.lower():
            return row[k].strip().lower()
    return ""

def _get_column_by_substr(row: dict, substr: str) -> str:
    """Return the first cell whose column name contains substr (case-insensitive)."""
    for k in row:
        if substr in k.lower():
            return (row[k] or "").strip()
    return ""

def filter_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if include_exclude_key(r) == "y"]

def is_doublette(row: dict) -> bool:
    return "doublette" in _get_column_by_substr(row, "further_notes").lower()

def drop_doublettes(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not is_doublette(r)]

def _parse_year(row: dict) -> Optional[int]:
    year_raw = row.get("pub_year", "").strip()
    try:
        return int(float(year_raw)) if year_raw else None
    except ValueError:
        return None

def parse_words(cell: str) -> list[str]:
    if not cell:
        return []
    return [p.strip() for p in re.split(r"[,;\n\r]+", cell) if p.strip()]

# ---------------------------------------------------------------------------
# 2.  KWIC analysis helpers
# ---------------------------------------------------------------------------

def _cooc_counter(kwic_cell: str) -> Counter:
    """Raw lemma Counter for a KWIC cell (no truncation)."""
    if not kwic_cell:
        return Counter()
    hit_words = {m.group(1).lower() for m in _HIT_RE.finditer(kwic_cell)}
    plain     = _HIT_RE.sub("", kwic_cell)
    counter: Counter = Counter()
    for line in plain.split("\n"):
        line = line.strip()
        if not line or _LABEL_RE.match(line):
            continue
        for word in _WORD_RE.findall(line.lower()):
            if word in hit_words:
                continue
            if word in STOPWORDS or len(word) < 3:
                continue
            lemma = lemmatize(word)
            if lemma in STOPWORDS or len(lemma) < 3:
                continue
            counter[lemma] += 1
    return counter

def _colloc_counter(kwic_cell: str, window: int = COLLOC_WIN) -> Counter:
    """Raw immediate-neighbour Counter for a KWIC cell (no truncation)."""
    if not kwic_cell:
        return Counter()
    counter: Counter = Counter()
    for line in kwic_cell.split("\n"):
        line = line.strip()
        if not line or _LABEL_RE.match(line):
            continue
        parts  = re.split(r"\*\*\w+\*\*", line)
        n_hits = len(_HIT_RE.findall(line))
        for idx in range(n_hits):
            left_words  = _WORD_RE.findall(parts[idx].lower())   if idx < len(parts)   else []
            right_words = _WORD_RE.findall(parts[idx+1].lower()) if idx+1 < len(parts) else []
            for w in left_words[-window:] + right_words[:window]:
                if w in STOPWORDS or len(w) < 3:
                    continue
                lemma = lemmatize(w)
                if lemma in STOPWORDS or len(lemma) < 3:
                    continue
                counter[lemma] += 1
    return counter

def _iter_kwic_blocks(kwic_cell: str, query_words: list[str],
                      canonicalise=None) -> list[tuple[str, str]]:
    """
    Return (canonical_key, block_text) pairs by splitting a multi-word KWIC cell
    on its label lines.  Single-word cells return one pair with the sole canonical key.
    """
    if not kwic_cell or not query_words:
        return []
    canon           = canonicalise if canonicalise else (lambda w: w.lower())
    label_to_canon  = {w.lower(): canon(w) for w in query_words}
    canonical_words = list(dict.fromkeys(label_to_canon.values()))

    if len(canonical_words) == 1:
        return [(canonical_words[0], kwic_cell)]

    results: list[tuple[str, str]] = []
    current_key  = None
    block_lines: list[str] = []

    def flush():
        if current_key and block_lines:
            results.append((current_key, "\n".join(block_lines)))

    for line in kwic_cell.split("\n"):
        stripped = line.strip()
        matched_key = None
        if _LABEL_RE.match(stripped):
            candidate = stripped.rstrip(":").lower()
            if candidate in label_to_canon:
                matched_key = label_to_canon[candidate]
        if matched_key:
            flush()
            current_key = matched_key
            block_lines = []
        else:
            block_lines.append(line)
    flush()
    return results

def _accumulate_kwic_cooc(kwic_cell: str, query_words: list[str],
                           cooc_acc: defaultdict, colloc_acc: defaultdict,
                           year: Optional[int], year_cooc_acc: defaultdict,
                           canonicalise=None) -> None:
    """
    Single pass over a KWIC cell that populates three accumulators:
      cooc_acc      – {canonical_word -> Counter(lemma -> total_count)}
      colloc_acc    – {canonical_word -> Counter(lemma -> total_count)}  (±window)
      year_cooc_acc – {year -> {canonical_word -> Counter(lemma -> count)}}
    """
    for key, block in _iter_kwic_blocks(kwic_cell, query_words, canonicalise):
        for lemma, cnt in _cooc_counter(block).items():
            cooc_acc[key][lemma] += cnt
            if year is not None:
                year_cooc_acc[year][key][lemma] += cnt
        for lemma, cnt in _colloc_counter(block).items():
            colloc_acc[key][lemma] += cnt

# ---------------------------------------------------------------------------
# 3.  Aggregate statistics
# ---------------------------------------------------------------------------

def _accumulate_textile_cooc_from_clean(
        clean_rows: list[dict],
        cooc_acc: defaultdict, colloc_acc: defaultdict,
        year_cooc_acc: defaultdict,
        cooc_by_cat: defaultdict, colloc_by_cat: defaultdict) -> None:
    """
    Populate the textile co-occurrence / collocation accumulators from the
    clean, one-row-per-occurrence table (CLEAN_CSV_PATH). Each row already
    represents exactly one textile-word hit with its own concordance
    snippet, so — unlike the FULL table's multi-word KWIC cells — no
    label-line block splitting is needed here.
    """
    for row in clean_rows:
        word = row.get("textile_words", "").strip()
        if not word:
            continue
        canon = textile_canonical(word)
        kwic  = row.get("kwic_textile", "") or ""
        year  = _parse_year(row)
        usage_raw = _get_column_by_substr(row, "usage_textile")
        cats = parse_usage_categories(usage_raw, TEXTILE_USAGE_VARIANTS, TEXTILE_USAGE_CATEGORIES)

        cooc_counts   = _cooc_counter(kwic)
        colloc_counts = _colloc_counter(kwic)

        for lemma, cnt in cooc_counts.items():
            cooc_acc[canon][lemma] += cnt
            if year is not None:
                year_cooc_acc[year][canon][lemma] += cnt
        for lemma, cnt in colloc_counts.items():
            colloc_acc[canon][lemma] += cnt

        # Per-category co-occurrence and collocation, once per recognised
        # category the occurrence belongs to
        for cat in cats:
            for lemma, cnt in cooc_counts.items():
                cooc_by_cat[cat][canon][lemma] += cnt
            for lemma, cnt in colloc_counts.items():
                colloc_by_cat[cat][canon][lemma] += cnt

def build_stats(rows: list[dict], all_rows: Optional[list[dict]] = None,
                 clean_rows: Optional[list[dict]] = None) -> dict:
    # rows        – texts with include_exclude == "y"; everything except the
    #               all-category/group stats below is computed from this set
    # all_rows    – every loaded row, "y" and "n" alike; falls back to `rows`
    #               if not given (keeps standalone calls to build_stats working)
    # clean_rows  – rows from the clean, one-occurrence-per-row textile KWIC
    #               table (CLEAN_CSV_PATH); drives the textile co-occurrence
    #               and collocation/concordance stats. Falls back to `rows`
    #               (the FULL table, with its multi-word KWIC blocks) if not
    #               given, so standalone calls to build_stats keep working.
    years         : list[int]   = []
    sources       : Counter     = Counter()
    source_years  : defaultdict = defaultdict(list)
    source_textile: defaultdict = defaultdict(int)
    textile_freq  : Counter     = Counter()
    year_textile  : defaultdict = defaultdict(Counter)

    year_text_counts: Counter = Counter()  # texts per year, used to normalise hit rates

    textile_cooc_acc  : defaultdict = defaultdict(Counter)
    textile_colloc_acc: defaultdict = defaultdict(Counter)
    year_cooc_textile : defaultdict = defaultdict(lambda: defaultdict(Counter))

    # Textile usage-category counters (notes_metaphorical_usage_textile).
    # Every count below is per category per text: within a single text a category
    # is counted at most once, however many times its words recur. The seven-category
    # totals feed the Usage section; the per-word split feeds the Words section.
    textile_usage_counts_all : Counter     = Counter()   # all seven categories
    year_textile_usage_all   : defaultdict = defaultdict(Counter)
    word_textile_usage       : defaultdict = defaultdict(Counter)  # canonical word -> category -> texts
    source_textile_by_cat    : defaultdict = defaultdict(lambda: defaultdict(int))

    # Group totals for the Overview chart. Metaphorical counts each text once if it
    # has any metaphor use (Textile and/or General Metaphor collapse to one, so it
    # equals textile_metaphor_texts). Non-metaphorical counts each text once per
    # non-metaphorical category it carries (the five categories summed).
    textile_group_counts     : Counter     = Counter()
    # Distinct metaphorical texts (headline stat) = the Metaphorical group above.
    textile_metaphor_texts   : int         = 0

    # Construction is a plain presence flag: one entry per text that contains a
    # construction word, with no sub-categories.
    const_usage_counts    : Counter     = Counter()
    year_const_usage      : defaultdict = defaultdict(Counter)
    source_const_texts    : defaultdict = defaultdict(int)
    construction_texts    : int = 0

    # Per-category co-occurrence and collocation accumulators (textile):
    # {t_cat -> {canonical_word -> Counter(lemma -> count)}}
    textile_cooc_by_cat  : defaultdict = defaultdict(lambda: defaultdict(Counter))
    textile_colloc_by_cat: defaultdict = defaultdict(lambda: defaultdict(Counter))

    # Texts per year split by textile metaphor category, for the normalised
    # "Distribution Over Time" chart. Counts texts, not word tokens.
    year_textile_by_cat: defaultdict = defaultdict(Counter)

    for row in rows:
        source   = row.get("journal_title", "").strip()
        year     = _parse_year(row)

        if year:
            years.append(year)
            year_text_counts[year] += 1
        if source:
            sources[source] += 1
            if year:
                source_years[source].append(year)

        t_words = parse_words(row.get("textile_words", ""))
        c_words = parse_words(row.get("construction_words", ""))

        # Parse the textile usage-category cell (may hold several comma-separated values).
        # Construction is now a plain presence flag: a text counts as "construction"
        # whenever it contains a construction word, with no usage sub-categories.
        t_usage_raw = _get_column_by_substr(row, "usage_textile")
        t_cats = parse_usage_categories(t_usage_raw, TEXTILE_USAGE_VARIANTS, TEXTILE_USAGE_CATEGORIES)
        c_cats = {"construction"} if c_words else set()

        for w in t_words:
            w = textile_canonical(w)
            textile_freq[w] += 1
            if year:
                year_textile[year][w] += 1
            if source:
                source_textile[source] += 1

        for cat in t_cats:
            if year:
                year_textile_by_cat[year][TEXTILE_USAGE_LABELS[cat]] += 1
            if source:
                source_textile_by_cat[source][TEXTILE_USAGE_LABELS[cat]] += 1

        # Construction text-level count per source (once per text, only if c_words present)
        if c_words and source:
            source_const_texts[source] += 1

        # Total construction texts, once per text — same condition as the
        # per-source count above (just c_words present, no other requirement)
        if c_words:
            construction_texts += 1

        # Construction presence total and temporal breakdown (one entry per text
        # that contains a construction word; identical population to the count above)
        for cat in c_cats:
            const_usage_counts[cat] += 1
            if year:
                year_const_usage[year][cat] += 1

    # Textile co-occurrence / collocation (Kookurrenz / Konkordanz) come from
    # the clean, one-row-per-occurrence table rather than `rows` (FULL).
    _accumulate_textile_cooc_from_clean(
        clean_rows if clean_rows is not None else rows,
        textile_cooc_acc, textile_colloc_acc, year_cooc_textile,
        textile_cooc_by_cat, textile_colloc_by_cat)

    # All-category and include/exclude group statistics draw on the full,
    # unfiltered row set — not just the "y" rows in `rows` above — because
    # the excluded usage categories (Tech Jargon, Textile Reference, Verb,
    # Unrelated, Other Language) live mostly in rows where include_exclude
    # is "n" and would otherwise never be counted at all.
    #
    # Counting is done per category per text, so the source is grouped by id
    # first (one entry per text, categories de-duplicated) and only then tallied.
    # This stays correct whether the FULL table is one row per text or is later
    # expanded to one row per occurrence. Metaphor categories are counted only
    # for included texts; the non-metaphor categories are counted everywhere.
    category_source_rows = all_rows if all_rows is not None else rows
    category_years: list[int] = []

    # Pass 1 — aggregate each text (id) into its year, include flag, the set of
    # categories it carries, and the (canonical word, metaphor category) pairs.
    id_year     : dict = {}
    id_included : dict = {}
    id_cats     : defaultdict = defaultdict(set)
    id_wordcats : defaultdict = defaultdict(set)
    for row in category_source_rows:
        cid  = row.get("id", "")
        year = _parse_year(row)
        if year:
            category_years.append(year)
            id_year[cid] = year
        else:
            id_year.setdefault(cid, None)
        if include_exclude_key(row) == "y":
            id_included[cid] = True
        else:
            id_included.setdefault(cid, False)
        t_usage_seq = parse_usage_sequence(_get_column_by_substr(row, "usage_textile"), TEXTILE_USAGE_VARIANTS)
        t_words_seq = parse_words(row.get("textile_words", ""))
        for cat in t_usage_seq:
            if cat is not None:
                id_cats[cid].add(cat)
        for word, cat in zip(t_words_seq, t_usage_seq):
            if cat in TEXTILE_USAGE_CATEGORIES:
                id_wordcats[cid].add((textile_canonical(word), cat))

    # Pass 2 — tally one entry per text per category, and assign each text to a
    # single group so Metaphorical/Non-metaphorical partition the corpus.
    for cid, cats in id_cats.items():
        year     = id_year.get(cid)
        included = id_included.get(cid, False)
        is_metaphorical_text = included and bool(cats & TEXTILE_USAGE_CATEGORIES)
        for cat in cats:
            is_metaphor = cat in TEXTILE_USAGE_CATEGORIES
            if is_metaphor and not included:
                continue
            if cat in ALL_TEXTILE_USAGE_CATEGORIES:
                textile_usage_counts_all[cat] += 1
                if not is_metaphor:
                    textile_group_counts["Non-metaphorical"] += 1
                if year:
                    year_textile_usage_all[year][cat] += 1
        if is_metaphorical_text:
            textile_group_counts["Metaphorical"] += 1
            textile_metaphor_texts += 1
            for word, cat in id_wordcats[cid]:
                word_textile_usage[word][cat] += 1

    year_range  = (min(years), max(years)) if years else (None, None)
    year_counts : Counter = Counter(years)
    all_years   = list(range(year_range[0], year_range[1] + 1)) if year_range[0] else []
    # Widen the year axis if the unfiltered rows cover years outside the
    # "y"-only range, so the all-category/group temporal charts aren't cut off
    if category_years:
        full_min = min(years + category_years) if years else min(category_years)
        full_max = max(years + category_years) if years else max(category_years)
        all_years = list(range(full_min, full_max + 1))

    # Normalise temporal hit counts per text in that year
    def normalise_year_counts(year_word_dict: defaultdict) -> dict:
        result = {}
        for y, word_counts in year_word_dict.items():
            n = year_text_counts.get(y, 1)
            result[y] = {w: round(c / n, 3) for w, c in word_counts.items()}
        return result

    # Normalise temporal co-occurrence by text count per year
    def normalise_year_cooc(acc: defaultdict) -> dict:
        result: dict = {}
        for year, word_dict in acc.items():
            n = year_text_counts.get(year, 1)
            for word, counter in word_dict.items():
                word_entry = result.setdefault(word, {})
                for lemma, cnt in counter.items():
                    word_entry.setdefault(lemma, {})[year] = round(cnt / n, 3)
        return result

    def top_cooc(acc: defaultdict) -> dict:
        return {w: acc[w].most_common(TOP_N_COOC) for w in acc}

    def top_colloc(acc: defaultdict) -> dict:
        return {w: acc[w].most_common(TOP_N_COLLOC) for w in acc}

    return {
        "textile_metaphor_texts": textile_metaphor_texts,
        "year_range"       : list(year_range),
        "sources"          : sources.most_common(),
        "source_years"     : {s: [min(y), max(y)] for s, y in source_years.items()},
        "source_textile"   : dict(source_textile),
        "all_years"        : all_years,
        "year_counts"      : {y: year_counts.get(y, 0) for y in all_years},
        "year_text_counts" : {y: year_text_counts.get(y, 0) for y in all_years},
        "textile_freq"     : textile_freq.most_common(),
        "year_textile"     : normalise_year_counts(year_textile),
        "textile_cooc"     : top_cooc(textile_cooc_acc),
        "textile_colloc"   : top_colloc(textile_colloc_acc),
        "year_cooc_textile": normalise_year_cooc(year_cooc_textile),
        "textile_words"    : [w for w, _ in textile_freq.most_common()],
        "textile_variants" : TEXTILE_VARIANTS,
        # --- Textile usage: per-word metaphor split (Words section) ---
        # {canonical_word -> [[display_label, count], ...]}
        "word_textile_usage"        : {
            w: [[TEXTILE_USAGE_LABELS.get(cat, cat), cnt]
                for cat, cnt in counter.most_common()]
            for w, counter in word_textile_usage.items()
        },
        # Textile texts per source split by category: {source -> {display_label -> count}}
        "source_textile_by_cat"  : {
            s: dict(cats) for s, cats in source_textile_by_cat.items()
        },
        # Construction texts per source (text-level, for the source chart)
        "source_const_texts"     : dict(source_const_texts),
        # Total construction texts corpus-wide, text-level (consistent with source_const_texts)
        "construction_texts"     : construction_texts,
        # Display-ordered category labels for JS iteration
        "textile_usage_category_labels": [
            TEXTILE_USAGE_LABELS[k] for k in ["textile metaphor", "general metaphor"]
            if k in TEXTILE_USAGE_LABELS
        ],
        # --- Textile usage categories, all seven (metaphorical + non-metaphorical + other) ---
        # Overall count per category, across all categories (not just the 2 focus ones)
        "textile_usage_counts_all"      : [
            [TEXTILE_USAGE_LABELS.get(k, k), v]
            for k, v in textile_usage_counts_all.most_common()
        ],
        # Display-ordered labels for all 7 categories, include-categories first
        "textile_usage_category_labels_all": [
            TEXTILE_USAGE_LABELS[k] for k in
            ["textile metaphor", "general metaphor",
             "tech jargon", "textile reference", "verb",
             "unrelated", "other language"]
            if k in TEXTILE_USAGE_LABELS
        ],
        # Temporal: {year -> {display_label -> count}}, across all 7 categories
        "year_textile_usage_all"        : {
            y: {TEXTILE_USAGE_LABELS.get(cat, cat): cnt
                for cat, cnt in cats.items()}
            for y, cats in year_textile_usage_all.items()
        },
        # --- Metaphorical vs. Non-metaphorical group totals (Overview) ---
        # The seven category counts summed into two groups (usage assignments, not texts).
        "textile_usage_group_counts"    : [
            [g, textile_group_counts.get(g, 0)] for g in TEXTILE_USAGE_GROUP_LABELS
        ],
        "textile_usage_group_labels"    : TEXTILE_USAGE_GROUP_LABELS,
        # --- Construction presence (single general category) ---
        "const_usage_counts"        : [
            [CONST_USAGE_LABELS.get(k, k), v]
            for k, v in const_usage_counts.most_common()
        ],
        "year_const_usage"          : {
            y: {CONST_USAGE_LABELS.get(cat, cat): cnt
                for cat, cnt in cats.items()}
            for y, cats in year_const_usage.items()
        },
        "const_usage_category_labels": [
            CONST_USAGE_LABELS["construction"]
        ],
        # Texts per year split by textile metaphor category (Distribution Over Time)
        # {year -> {display_label -> count}}
        "year_textile_by_cat": {
            y: dict(cat_counter) for y, cat_counter in year_textile_by_cat.items()
        },
        # --- Per-category co-occurrence and collocation (textile) ---
        # {category_key -> {word -> [[lemma, count], ...]}}
        "textile_cooc_by_cat"  : {
            TEXTILE_USAGE_LABELS.get(cat, cat): top_cooc(acc)
            for cat, acc in textile_cooc_by_cat.items()
        },
        "textile_colloc_by_cat": {
            TEXTILE_USAGE_LABELS.get(cat, cat): top_colloc(acc)
            for cat, acc in textile_colloc_by_cat.items()
        },
    }

# ---------------------------------------------------------------------------
# 4.  HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Weaving DH — Corpus Dashboard</title>
<link rel="icon" type="image/jpeg" href="https://ids.si.edu/ids/deliveryService?id=NMAH-AHB2019q157831-000001&max=64">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Serif+Display&family=JetBrains+Mono:wght@400&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --black:  #0a0a0a;
    --white:  #f8f8f6;
    --grey-1: #1c1c1c;
    --grey-2: #3a3a3a;
    --grey-3: #7a7a7a;
    --grey-4: #c8c8c8;
    --grey-5: #ebebeb;
    --rule:   1px solid #d0d0d0;
    --c-textile: #2563a8;
    --c-const:   #b5451b;
    --c-textile-metaphor: #2563a8;
    --c-general-metaphor: #93b8e0;
    --font-display: 'DM Serif Display', Georgia, serif;
    --font-body:    'DM Sans', system-ui, sans-serif;
    --font-mono:    'JetBrains Mono', 'Fira Mono', monospace;
  }

  html { scroll-behavior: smooth; }
  body { background: var(--white); color: var(--black); font-family: var(--font-body); font-size: 15px; line-height: 1.6; }
  .page-wrap { max-width: 1200px; margin: 0 auto; padding: 0 2rem; }

  header { border-bottom: 2px solid var(--black); padding: 3rem 0 2rem; }
  .header-inner { display: flex; align-items: flex-start; justify-content: space-between; gap: 2rem; }
  .site-title { font-family: var(--font-display); font-size: clamp(2rem,5vw,3.2rem); letter-spacing: -0.02em; line-height: 1.1; }
  .header-logo { width: 120px; height: 120px; object-fit: cover; flex-shrink: 0; border: var(--rule); }

  nav { border-bottom: var(--rule); padding: 0.75rem 0; position: sticky; top: 0; background: var(--white); z-index: 100; }
  nav ul { display: flex; gap: 2rem; list-style: none; flex-wrap: wrap; }
  nav a { font-size: 0.8rem; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; color: var(--grey-2); text-decoration: none; }
  nav a:hover { color: var(--black); }

  section { padding: 3.5rem 0; border-bottom: var(--rule); }
  section:last-of-type { border-bottom: none; }
  .section-label { font-size: 0.7rem; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; color: var(--grey-3); margin-bottom: 0.5rem; }
  .section-title { font-family: var(--font-display); font-size: clamp(1.4rem,3vw,2rem); letter-spacing: -0.01em; margin-bottom: 2rem; }
  .subsection-title { font-size: 0.75rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: var(--grey-2); margin: 2.5rem 0 1rem; border-top: var(--rule); padding-top: 1.5rem; }

  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(160px,1fr)); border: 2px solid var(--black); }
  .stat-card { padding: 1.5rem; border-right: var(--rule); }
  .stat-card:last-child { border-right: none; }
  .stat-number { font-family: var(--font-display); font-size: 2.8rem; line-height: 1; letter-spacing: -0.03em; }
  .stat-label { font-size: 0.75rem; color: var(--grey-3); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.4rem; }

  .source-table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
  .source-table th { text-align: left; font-weight: 500; font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--grey-3); padding: 0.5rem 1rem 0.5rem 0; border-bottom: var(--rule); }
  .source-table td { padding: 0.6rem 1rem 0.6rem 0; border-bottom: var(--rule); color: var(--grey-1); }
  .source-table tr:last-child td { border-bottom: none; }

  .chart-wrap { position: relative; background: #fff; border: var(--rule); padding: 1.5rem; margin-bottom: 1.5rem; }
  .chart-title { font-size: 0.75rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: var(--grey-2); margin-bottom: 1rem; }
  .chart-note { font-size: 0.72rem; color: var(--grey-3); margin-top: 0.9rem; font-style: italic; line-height: 1.5; }

  .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
  @media (max-width: 700px) { .chart-row { grid-template-columns: 1fr; } }

  .tab-group { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1.25rem; }
  .tab-btn { padding: 0.3rem 0.8rem; font-size: 0.78rem; font-family: var(--font-mono); border: 1px solid var(--grey-4); background: transparent; cursor: pointer; color: var(--grey-2); transition: background 0.12s, color 0.12s; }
  .tab-btn:hover { background: var(--grey-5); }
  .tab-btn.active { background: var(--black); color: var(--white); border-color: var(--black); }

  .cooc-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(280px,1fr)); gap: 1.5rem; }
  .cooc-card { border: var(--rule); padding: 1.25rem; }
  .cooc-card-title { font-family: var(--font-mono); font-size: 0.8rem; color: var(--grey-2); margin-bottom: 0.25rem; padding-bottom: 0.5rem; border-bottom: var(--rule); }
  .cooc-card-variants { font-size: 0.7rem; color: var(--grey-3); margin-bottom: 0.9rem; font-style: italic; }
  .cooc-row { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }
  .cooc-word { width: 110px; flex-shrink: 0; color: var(--grey-1); font-family: var(--font-mono); font-size: 0.75rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .cooc-bar-wrap { flex: 1; background: var(--grey-5); height: 10px; }
  .cooc-bar { height: 100%; background: var(--black); transition: width 0.3s ease; }
  .cooc-count { width: 28px; text-align: right; color: var(--grey-3); font-size: 0.72rem; font-family: var(--font-mono); }


  .variants-note { font-size: 0.75rem; color: var(--grey-3); font-style: italic; margin-top: -1rem; margin-bottom: 1.5rem; }

  .legend-inline { display: flex; gap: 1.5rem; margin-bottom: 0.75rem; }
  .legend-item { display: flex; align-items: center; gap: 0.4rem; font-size: 0.75rem; color: var(--grey-2); }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }

  .tab-btn-swatch { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.3rem; vertical-align: middle; flex-shrink: 0; }
</style>
</head>
<body>
<div class="page-wrap">

<header>
  <div class="header-inner">
    <div><h1 class="site-title">Weaving DH</h1></div>
    <img class="header-logo"
         src="https://ids.si.edu/ids/deliveryService?id=NMAH-AHB2019q157831-000001&max=300"
         alt="Weaving DH logo">
  </div>
</header>

<nav>
  <ul>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#temporal">Over Time</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#vocabulary">Words</a></li>
    <li><a href="#sources">Sources</a></li>
    <li><a href="#cooccurrence">Context</a></li>
    <li><a href="#construction">Construction</a></li>
    <li><a href="#methodology">Notes</a></li>
  </ul>
</nav>

<section id="overview">
  <p class="section-label">Corpus</p>
  <h2 class="section-title">Overview</h2>
  <div class="stats-grid">
    <div class="stat-card"><div class="stat-number" id="stat-sources">—</div><div class="stat-label">Sources</div></div>
    <div class="stat-card"><div class="stat-number" id="stat-years">—</div><div class="stat-label">Year range</div></div>
    <div class="stat-card"><div class="stat-number" id="stat-textile-metaphor">—</div><div class="stat-label">Metaphorical Textile texts</div></div>
    <div class="stat-card"><div class="stat-number" id="stat-const">—</div><div class="stat-label">Construction texts</div></div>
  </div>
  <br>
  <p class="subsection-title">Metaphorical vs. Non-metaphorical</p>
  <div class="legend-inline" id="legend-textile-usage-group"></div>
  <div class="chart-wrap">
    <canvas id="chart-textile-usage-group-total" height="80"></canvas>
    <p class="chart-note">Metaphorical counts each text once if it has any metaphor use — multiple metaphors in a text, including both Textile and General, still count once (so this equals the metaphorical-texts figure above). Non-metaphorical counts each text once per non-metaphorical category it carries (Tech Jargon, Textile Reference, Verb, Unrelated, Other Language).</p>
  </div>
  <p class="chart-title" style="margin-top:1.75rem;">Sources in corpus</p>
  <table class="source-table">
    <thead><tr><th>Source</th><th>Texts</th></tr></thead>
    <tbody id="source-tbody"></tbody>
  </table>
</section>

<section id="temporal">
  <p class="section-label">Chronology</p>
  <h2 class="section-title">Distribution Over Time</h2>
  <p style="font-size:0.85rem;color:var(--grey-2);margin-bottom:1.5rem;">
    How the corpus and its textile-metaphor use spread across publication years. The first
    chart is the raw number of included texts per year; the two below show, per year, the
    rate at which textile words were used — split by metaphor category and by individual
    word. The lower charts are normalised by the number of texts that year, so a tall bar
    means a high rate, not simply more texts.
  </p>
  <div class="chart-wrap">
    <p class="chart-title">Texts per year</p>
    <canvas id="chart-year-total" height="80"></canvas>
    <p class="chart-note">Number of included texts published each year.</p>
  </div>

  <p class="subsection-title">Textile word hits per year</p>
  <div class="legend-inline" id="legend-temporal-textile"></div>
  <div class="chart-row">
    <div class="chart-wrap">
      <p class="chart-title">By metaphor category (normalised)</p>
      <canvas id="chart-year-textile-cat" height="160"></canvas>
      <p class="chart-note">Texts per year in which a textile word appeared, split by usage category. Normalised by annual text count.</p>
    </div>
    <div class="chart-wrap">
      <p class="chart-title">By word (normalised)</p>
      <canvas id="chart-year-textile" height="160"></canvas>
      <p class="chart-note">Hits per text per year, normalised by annual text count to account for uneven corpus distribution across years.</p>
    </div>
  </div>
</section>

<section id="usage">
  <p class="section-label">Classification</p>
  <h2 class="section-title">How Textile Words Are Used</h2>
  <p style="font-size:0.85rem;color:var(--grey-2);margin-bottom:1.5rem;">
    Breakdown of textile-word usage into its seven categories, with the two metaphor
    categories highlighted. Counts draw on the full annotated corpus
    (<code>include_exclude</code> = y and n), since the non-metaphorical categories occur
    mostly in excluded texts. Each text is counted once per category, so a text tagged both
    Textile and General Metaphor appears in both bars — 353 distinct texts are metaphorical
    (15 carry both metaphor types).
  </p>
  <div class="legend-inline" id="legend-textile-usage-all"></div>
  <div class="chart-row">
    <div class="chart-wrap">
      <p class="chart-title">Texts per category</p>
      <canvas id="chart-textile-usage-all-total" height="120"></canvas>
      <p class="chart-note">All seven categories side by side, each text counted once per assigned category.</p>
    </div>
    <div class="chart-wrap">
      <p class="chart-title">Temporal distribution</p>
      <canvas id="chart-textile-usage-all-year" height="120"></canvas>
      <p class="chart-note">Absolute counts per year, stacked across all seven categories.</p>
    </div>
  </div>
</section>

<section id="vocabulary">
  <p class="section-label">Vocabulary</p>
  <h2 class="section-title">Textile Words</h2>
  <p class="variants-note" id="textile-variants-note"></p>
  <div class="chart-row">
    <div class="chart-wrap">
      <p class="chart-title">Frequency across corpus</p>
      <canvas id="chart-textile-freq"></canvas>
      <p class="chart-note">Search used unlemmatised surface forms; morphological variants (shown above) were grouped manually in post-processing.</p>
    </div>
    <div class="chart-wrap">
      <p class="chart-title">Temporal distribution — select word</p>
      <div class="tab-group" id="tabs-textile"></div>
      <canvas id="chart-textile-word-time" height="160"></canvas>
      <p class="chart-note">Normalised hits per text per year.</p>
    </div>
  </div>

  <p class="subsection-title">Metaphor split per word</p>
  <p style="font-size:0.85rem;color:var(--grey-2);margin-bottom:1rem;">How the <em>Textile Metaphor</em> vs. <em>General Metaphor</em> categories distribute across individual textile words.</p>
  <div class="legend-inline" id="legend-textile-usage"></div>
  <div class="chart-wrap">
    <canvas id="chart-textile-usage-per-word" height="160"></canvas>
    <p class="chart-note">Stacked counts per canonical textile word, sorted by Textile Metaphor frequency (descending). A text using two different textile words appears under each, so per-word totals exceed the distinct-text counts in the Usage section.</p>
  </div>
</section>

<section id="sources">
  <p class="section-label">Sources</p>
  <h2 class="section-title">By Source</h2>
  <p class="subsection-title">Textile Metaphor Texts per Source — by Category</p>
  <div class="legend-inline" id="legend-source-textile"></div>
  <div class="chart-wrap">
    <canvas id="chart-source-ratio" height="120"></canvas>
    <p class="chart-note">Texts per source in which a textile metaphor word appeared, stacked by usage category (Textile Metaphor / General Metaphor). Each text is counted once per recognised category.</p>
  </div>
</section>

<section id="cooccurrence">
  <p class="section-label">Context</p>
  <h2 class="section-title">Co-occurrence &amp; Collocation</h2>

  <p class="subsection-title">Top 5 Co-occurring Words in KWIC Context</p>
  <p style="font-size:0.85rem;color:var(--grey-2);margin-bottom:1rem;">Most frequent lemmatised content words within the ±15-token KWIC window. Co-occurrence analysis uses WordNet noun lemmatization, which differs from the PorterStemmer used at KWIC search time.</p>
  <p class="chart-title" style="margin-bottom:0.75rem">Textile words</p>
  <div class="tab-group" id="tabs-cooc-textile-cat"></div>
  <div class="cooc-grid" id="cooc-textile"></div>

  <p class="subsection-title">Immediate Collocations (±2 tokens)</p>
  <p style="font-size:0.85rem;color:var(--grey-2);margin-bottom:1rem;">Words appearing directly adjacent to the hit token, revealing typical phrasal patterns (e.g. <em>building a corpus</em>, <em>weaving together</em>). Left and right positions are pooled.</p>
  <p class="chart-title" style="margin-bottom:0.75rem">Textile words</p>
  <div class="tab-group" id="tabs-colloc-textile-cat"></div>
  <div class="cooc-grid" id="colloc-textile"></div>

  <p class="subsection-title">Temporal Co-occurrence Trend</p>
  <p style="font-size:0.85rem;color:var(--grey-2);margin-bottom:1.5rem;">Co-occurrence frequency of a selected term alongside a given metaphor word, normalised per text per year.</p>
  <div class="chart-row">
    <div class="chart-wrap">
      <p class="chart-title">Textile — select word, then co-occurring term</p>
      <div class="tab-group" id="tabs-trend-textile-word"></div>
      <div class="tab-group" id="tabs-trend-textile-lemma"></div>
      <canvas id="chart-trend-textile" height="160"></canvas>
    </div>
  </div>
</section>

<section id="construction">
  <p class="section-label">Comparison</p>
  <h2 class="section-title">Construction Vocabulary</h2>
  <p style="font-size:0.85rem;color:var(--grey-2);margin-bottom:1.5rem;">Construction words (build, dig, mine) are tracked only as a coarse comparison — whether a text uses one at all, with no usage sub-categories.</p>
  <div class="chart-row">
    <div class="chart-wrap">
      <p class="chart-title">Texts total</p>
      <canvas id="chart-const-usage-total" height="120"></canvas>
      <p class="chart-note">Number of included texts that contain at least one construction word.</p>
    </div>
    <div class="chart-wrap">
      <p class="chart-title">Temporal distribution</p>
      <canvas id="chart-const-usage-year" height="120"></canvas>
      <p class="chart-note">Construction texts per year (absolute counts).</p>
    </div>
  </div>
  <p class="subsection-title">Construction Texts per Source</p>
  <div class="chart-wrap">
    <canvas id="chart-source-const" height="80"></canvas>
    <p class="chart-note">Number of included texts per source that contain at least one construction word. Text-level count, not token count.</p>
  </div>
</section>

<section id="methodology">
  <h2 class="section-title">Notes</h2>
  <p class="chart-note">
    Corpus comprises all journal articles in which at least one textile metaphor word was identified and manually confirmed as metaphorical. Texts were fully extracted including abstracts; OCR quality was not manually verified and fuzzy OCR artefacts may affect word counts. Include/exclude decisions were made by a single annotator. Search used unlemmatised surface forms; morphological variants were added manually where relevant. Rows flagged as doublettes are excluded from all counts.
  </p>
</section>

</div>

<script>
const DATA = __DATA_PLACEHOLDER__;

const PALETTE = [
  '#2563a8','#b5451b','#2a7a4f','#7c3d99',
  '#b08a1e','#3a8a8a','#c4565e','#5a6e2a',
  '#1a5c8a','#8a3a1a','#1a5a3a','#5a2a7a',
];
function color(i){ return PALETTE[i % PALETTE.length]; }

Chart.defaults.font.family = "'DM Sans', system-ui, sans-serif";
Chart.defaults.font.size   = 11;
Chart.defaults.color       = '#7a7a7a';
const GRID  = { color: '#ebebeb', drawBorder: false };
const TICKS = { color: '#7a7a7a' };

// ── Usage category colour maps (defined early — referenced throughout) ─
const TEXTILE_USAGE_COLORS = {
  'Textile Metaphor': '#2563a8',
  'General Metaphor': '#93b8e0',
};
const CONST_USAGE_COLORS = {
  'Construction':      '#b5451b',
};
// All seven textile usage categories: blue shades = Metaphorical,
// warm/grey shades = Non-metaphorical
const TEXTILE_ALL_CATEGORY_COLORS = {
  'Textile Metaphor':  '#2563a8',
  'General Metaphor':  '#93b8e0',
  'Tech Jargon':       '#b5451b',
  'Textile Reference': '#e08a6a',
  'Verb':              '#d9a441',
  'Unrelated':         '#9a9a9a',
  'Other Language':    '#c8c8c8',
};
// Group-level colours (Metaphorical / Non-metaphorical)
const TEXTILE_GROUP_COLORS = {
  'Metaphorical':     '#2563a8',
  'Non-metaphorical': '#b5451b',
};

// ── Overview ──────────────────────────────────────────────────────
document.getElementById('stat-sources').textContent = DATA.sources.length;
document.getElementById('stat-years').textContent   =
  DATA.year_range[0] ? `${DATA.year_range[0]}–${DATA.year_range[1]}` : '—';

// Textile Metaphor texts: rows tagged Textile Metaphor and/or General Metaphor,
// counted once per text (computed in Python, not re-derived here)
document.getElementById('stat-textile-metaphor').textContent = DATA.textile_metaphor_texts.toLocaleString();

// Construction texts: rows with at least one construction word, counted once
// per text — same definition used by the per-source breakdown below
document.getElementById('stat-const').textContent =
  DATA.construction_texts.toLocaleString();

const tbody = document.getElementById('source-tbody');
DATA.sources.forEach(([j,n]) => {
  const yr    = DATA.source_years[j];
  const yrStr = yr ? (yr[0]===yr[1] ? ` (${yr[0]})` : ` (${yr[0]}–${yr[1]})`) : '';
  const tr    = document.createElement('tr');
  tr.innerHTML = `<td>${j}${yrStr}</td><td>${n}</td>`;
  tbody.appendChild(tr);
});

// Variants notes
const textileVariantParts = Object.entries(DATA.textile_variants).map(([w,v])=>`${w} (incl. ${v})`);
document.getElementById('textile-variants-note').textContent =
  'Morphological variants included: ' + textileVariantParts.join('; ') + '.';

// ── Source charts ─────────────────────────────────────────────────
(function(){
  const sources    = DATA.sources.map(([s])=>s);
  const catLabels  = DATA.textile_usage_category_labels;

  // Legend for source textile chart
  const leg = document.getElementById('legend-source-textile');
  catLabels.forEach(label => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<div class="legend-dot" style="background:${TEXTILE_USAGE_COLORS[label]||'#ccc'}"></div>${label}`;
    leg.appendChild(item);
  });

  // Textile Metaphor texts per source — stacked by category
  new Chart(document.getElementById('chart-source-ratio'), {
    type: 'bar',
    data: {
      labels: sources,
      datasets: catLabels.map(label => ({
        label,
        data: sources.map(s => (DATA.source_textile_by_cat[s]||{})[label] || 0),
        backgroundColor: TEXTILE_USAGE_COLORS[label] || '#ccc',
        stack: 'a',
        borderWidth: 0,
      }))
    },
    options: {
      plugins:{ legend:{ display:false } },
      scales:{
        x:{ stacked:true, grid:GRID, ticks:{ ...TICKS, maxRotation:30, font:{size:10} } },
        y:{ stacked:true, grid:GRID, ticks:{ ...TICKS, stepSize:1 }, beginAtZero:true }
      }
    }
  });

  // Construction texts per source — text-level
  new Chart(document.getElementById('chart-source-const'), {
    type: 'bar',
    data: {
      labels: sources,
      datasets: [{
        data: sources.map(s => DATA.source_const_texts[s] || 0),
        backgroundColor: '#b5451b',
        borderWidth: 0,
      }]
    },
    options: {
      plugins:{ legend:{ display:false } },
      scales:{
        x:{ grid:GRID, ticks:{ ...TICKS, maxRotation:30, font:{size:10} } },
        y:{ grid:GRID, ticks:{ ...TICKS, stepSize:1 }, beginAtZero:true }
      }
    }
  });
})();

// ── Texts per year ────────────────────────────────────────────────
new Chart(document.getElementById('chart-year-total'), {
  type: 'bar',
  data: {
    labels: DATA.all_years,
    datasets: [{ data: DATA.all_years.map(y=>DATA.year_counts[y]||0), backgroundColor:'#0a0a0a', borderWidth:0 }]
  },
  options: { plugins:{legend:{display:false}}, scales:{ x:{grid:GRID,ticks:TICKS}, y:{grid:GRID,ticks:{...TICKS,stepSize:1},beginAtZero:true} } }
});

// ── Temporal legend (textile category split) ──────────────────────
(function(){
  const tLegend = document.getElementById('legend-temporal-textile');
  Object.entries(TEXTILE_USAGE_COLORS).forEach(([label, col]) => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<div class="legend-dot" style="background:${col}"></div>${label}`;
    tLegend.appendChild(item);
  });
})();

// ── Category-stacked temporal (normalised by text count per year) ──
function makeCatStacked(canvasId, yearByCat, catLabels, colorMap) {
  // Normalise: divide each count by texts that year
  new Chart(document.getElementById(canvasId), {
    type: 'bar',
    data: {
      labels: DATA.all_years,
      datasets: catLabels.map(label => ({
        label,
        data: DATA.all_years.map(y => {
          const n = DATA.year_text_counts[y] || 1;
          return parseFloat(((yearByCat[y]||{})[label] || 0) / n).toFixed(3);
        }),
        backgroundColor: colorMap[label] || '#c8c8c8',
        stack: 'a',
        borderWidth: 0,
      }))
    },
    options: {
      plugins:{ legend:{ display:false } },
      scales:{ x:{stacked:true,grid:GRID,ticks:TICKS}, y:{stacked:true,grid:GRID,ticks:TICKS,beginAtZero:true} }
    }
  });
}
makeCatStacked('chart-year-textile-cat', DATA.year_textile_by_cat,
               DATA.textile_usage_category_labels, TEXTILE_USAGE_COLORS);

// ── Stacked temporal by word (normalised) ─────────────────────────
function makeStacked(canvasId, words, yearData) {
  new Chart(document.getElementById(canvasId), {
    type: 'bar',
    data: {
      labels: DATA.all_years,
      datasets: words.map((w,i) => ({
        label: w, data: DATA.all_years.map(y=>(yearData[y]||{})[w]||0),
        backgroundColor: color(i), stack:'a', borderWidth:0,
      }))
    },
    options: {
      plugins:{ legend:{ position:'right', labels:{boxWidth:10,padding:8,font:{size:10}} } },
      scales:{ x:{stacked:true,grid:GRID,ticks:TICKS}, y:{stacked:true,grid:GRID,ticks:TICKS,beginAtZero:true} }
    }
  });
}
makeStacked('chart-year-textile', DATA.textile_words, DATA.year_textile);

// ── Frequency (horizontal bar) ────────────────────────────────────
function makeFreq(canvasId, freqData) {
  new Chart(document.getElementById(canvasId), {
    type: 'bar',
    data: {
      labels: freqData.map(([w])=>w),
      datasets: [{ data: freqData.map(([,n])=>n), backgroundColor: freqData.map((_,i)=>color(i)), borderWidth:0 }]
    },
    options: {
      indexAxis:'y',
      plugins:{legend:{display:false}},
      scales:{ x:{grid:GRID,ticks:TICKS,beginAtZero:true}, y:{grid:{display:false},ticks:{...TICKS,font:{family:"'JetBrains Mono',monospace",size:11}}} }
    }
  });
}
makeFreq('chart-textile-freq', DATA.textile_freq);

// ── Per-word temporal with tabs (normalised) ──────────────────────
function makeWordTime(canvasId, tabGroupId, yearData, words) {
  let chart = null;
  const tabs   = document.getElementById(tabGroupId);
  const canvas = document.getElementById(canvasId);
  function render(word, btn) {
    tabs.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    if (chart) chart.destroy();
    chart = new Chart(canvas, {
      type: 'bar',
      data: { labels: DATA.all_years, datasets: [{
        label: word, data: DATA.all_years.map(y=>(yearData[y]||{})[word]||0),
        backgroundColor:'#0a0a0a', borderWidth:0,
      }]},
      options:{ plugins:{legend:{display:false}}, scales:{ x:{grid:GRID,ticks:TICKS}, y:{grid:GRID,ticks:TICKS,beginAtZero:true} } }
    });
  }
  words.forEach((w,i) => {
    const btn = document.createElement('button');
    btn.className   = 'tab-btn';
    btn.textContent = w;
    btn.addEventListener('click', () => render(w, btn));
    tabs.appendChild(btn);
    if (i === 0) render(w, btn);
  });
}
makeWordTime('chart-textile-word-time','tabs-textile', DATA.year_textile, DATA.textile_words);

// ── Co-occurrence cards ───────────────────────────────────────────
function renderCooc(containerId, coocData, variants) {
  const container = document.getElementById(containerId);
  Object.entries(coocData).forEach(([word, pairs]) => {
    if (!pairs || pairs.length === 0) return;
    const maxVal = pairs[0][1];
    const card   = document.createElement('div');
    card.className = 'cooc-card';
    const variantNote = variants && variants[word]
      ? `<div class="cooc-card-variants">incl. ${variants[word]}</div>` : '';
    card.innerHTML = `<div class="cooc-card-title">${word}</div>${variantNote}`;
    pairs.forEach(([w,n]) => {
      const pct = maxVal > 0 ? Math.round((n/maxVal)*100) : 0;
      card.innerHTML += `
        <div class="cooc-row">
          <span class="cooc-word" title="${w}">${w}</span>
          <div class="cooc-bar-wrap"><div class="cooc-bar" style="width:${pct}%"></div></div>
          <span class="cooc-count">${n}</span>
        </div>`;
    });
    container.appendChild(card);
  });
}

// ── Temporal co-occurrence trend (normalised) ─────────────────────
function makeTrendChart(canvasId, wordTabId, lemmaTabId, yearCoocData, allYears) {
  let chart       = null;
  let currentWord = null;
  const wordTabs  = document.getElementById(wordTabId);
  const lemmaTabs = document.getElementById(lemmaTabId);
  const canvas    = document.getElementById(canvasId);

  function renderTrend(lemma, lemmaBtn) {
    lemmaTabs.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    lemmaBtn.classList.add('active');
    if (chart) chart.destroy();
    const wordData = (yearCoocData[currentWord] || {})[lemma] || {};
    chart = new Chart(canvas, {
      type: 'bar',
      data: { labels: allYears, datasets: [{
        label: lemma,
        data: allYears.map(y => wordData[y] || 0),
        backgroundColor: '#0a0a0a', borderWidth: 0,
      }]},
      options:{ plugins:{legend:{display:false}}, scales:{ x:{grid:GRID,ticks:TICKS}, y:{grid:GRID,ticks:TICKS,beginAtZero:true} } }
    });
  }

  function renderWord(word, wordBtn) {
    wordTabs.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    wordBtn.classList.add('active');
    currentWord = word;
    lemmaTabs.innerHTML = '';
    if (chart) { chart.destroy(); chart = null; }

    const lemmas = Object.keys((yearCoocData[word] || {}))
      .map(l => {
        const total = Object.values((yearCoocData[word][l] || {})).reduce((a,b)=>a+b,0);
        return [l, total];
      })
      .sort((a,b) => b[1]-a[1])
      .slice(0, 8)
      .map(([l]) => l);

    if (lemmas.length === 0) return;
    lemmas.forEach((l, i) => {
      const btn = document.createElement('button');
      btn.className   = 'tab-btn';
      btn.textContent = l;
      btn.addEventListener('click', () => renderTrend(l, btn));
      lemmaTabs.appendChild(btn);
      if (i === 0) renderTrend(l, btn);
    });
  }

  const words = Object.keys(yearCoocData);
  words.forEach((w, i) => {
    const btn = document.createElement('button');
    btn.className   = 'tab-btn';
    btn.textContent = w;
    btn.addEventListener('click', () => renderWord(w, btn));
    wordTabs.appendChild(btn);
    if (i === 0) renderWord(w, btn);
  });
}

makeTrendChart('chart-trend-textile','tabs-trend-textile-word','tabs-trend-textile-lemma',
               DATA.year_cooc_textile, DATA.all_years);

// ── Co-occurrence category filter tabs ────────────────────────────
// Renders category-tab switchers above the textile cooc/colloc grids.
// "All" shows DATA.textile_cooc / DATA.textile_colloc;
// per-category tabs show DATA.textile_cooc_by_cat[label] / textile_colloc_by_cat[label].
(function(){
  function makeCatTabs(tabGroupId, gridId, allData, byCategory, variants) {
    const tabs      = document.getElementById(tabGroupId);
    const container = document.getElementById(gridId);

    function renderCards(coocData) {
      container.innerHTML = '';
      Object.entries(coocData).forEach(([word, pairs]) => {
        if (!pairs || pairs.length === 0) return;
        const maxVal = pairs[0][1];
        const card   = document.createElement('div');
        card.className = 'cooc-card';
        const variantNote = variants && variants[word]
          ? `<div class="cooc-card-variants">incl. ${variants[word]}</div>` : '';
        card.innerHTML = `<div class="cooc-card-title">${word}</div>${variantNote}`;
        pairs.forEach(([w, n]) => {
          const pct = maxVal > 0 ? Math.round((n / maxVal) * 100) : 0;
          card.innerHTML += `
            <div class="cooc-row">
              <span class="cooc-word" title="${w}">${w}</span>
              <div class="cooc-bar-wrap"><div class="cooc-bar" style="width:${pct}%"></div></div>
              <span class="cooc-count">${n}</span>
            </div>`;
        });
        container.appendChild(card);
      });
    }

    function activate(btn, data) {
      tabs.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderCards(data);
    }

    // "All" tab
    const allBtn = document.createElement('button');
    allBtn.className   = 'tab-btn';
    allBtn.textContent = 'All';
    allBtn.addEventListener('click', () => activate(allBtn, allData));
    tabs.appendChild(allBtn);

    // Per-category tabs with colour swatch
    DATA.textile_usage_category_labels.forEach(label => {
      const catData = (byCategory[label]) || {};
      if (Object.keys(catData).length === 0) return;
      const col  = TEXTILE_USAGE_COLORS[label] || '#ccc';
      const btn  = document.createElement('button');
      btn.className = 'tab-btn';
      btn.innerHTML = `<span class="tab-btn-swatch" style="background:${col}"></span>${label}`;
      btn.addEventListener('click', () => activate(btn, catData));
      tabs.appendChild(btn);
    });

    // Activate "All" by default
    activate(allBtn, allData);
  }

  makeCatTabs('tabs-cooc-textile-cat',   'cooc-textile',   DATA.textile_cooc,   DATA.textile_cooc_by_cat,   DATA.textile_variants);
  makeCatTabs('tabs-colloc-textile-cat', 'colloc-textile', DATA.textile_colloc, DATA.textile_colloc_by_cat, DATA.textile_variants);
})();

// ── Usage category colour maps ─────────────────────────────────────
// (defined at top of script)

// Build legend rows for a usage-category section
function buildUsageLegend(containerId, colorMap) {
  const el = document.getElementById(containerId);
  Object.entries(colorMap).forEach(([label, col]) => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<div class="legend-dot" style="background:${col}"></div>${label}`;
    el.appendChild(item);
  });
}
buildUsageLegend('legend-textile-usage', TEXTILE_USAGE_COLORS);

// ── Usage category total bar chart (horizontal) ───────────────────
function makeUsageTotal(canvasId, usageCounts, colorMap) {
  if (!usageCounts || usageCounts.length === 0) return;
  new Chart(document.getElementById(canvasId), {
    type: 'bar',
    data: {
      labels: usageCounts.map(([l])=>l),
      datasets: [{
        data: usageCounts.map(([,n])=>n),
        backgroundColor: usageCounts.map(([l])=>colorMap[l]||'#c8c8c8'),
        borderWidth: 0,
      }]
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: GRID, ticks: TICKS, beginAtZero: true },
        y: { grid: { display: false }, ticks: { ...TICKS, font: { family:"'JetBrains Mono',monospace", size:11 } } }
      }
    }
  });
}
makeUsageTotal('chart-const-usage-total',   DATA.const_usage_counts,   CONST_USAGE_COLORS);

// ── Usage category temporal stacked bar chart ─────────────────────
function makeUsageYear(canvasId, yearUsage, categoryLabels, colorMap, allYears) {
  if (!categoryLabels || categoryLabels.length === 0) return;
  new Chart(document.getElementById(canvasId), {
    type: 'bar',
    data: {
      labels: allYears,
      datasets: categoryLabels.map(label => ({
        label,
        data: allYears.map(y => (yearUsage[y]||{})[label] || 0),
        backgroundColor: colorMap[label] || '#c8c8c8',
        stack: 'a',
        borderWidth: 0,
      }))
    },
    options: {
      plugins: { legend: { position:'right', labels:{ boxWidth:10, padding:8, font:{ size:10 } } } },
      scales: {
        x: { stacked: true, grid: GRID, ticks: TICKS },
        y: { stacked: true, grid: GRID, ticks: TICKS, beginAtZero: true }
      }
    }
  });
}
makeUsageYear('chart-const-usage-year',   DATA.year_const_usage,   DATA.const_usage_category_labels,
              CONST_USAGE_COLORS,   DATA.all_years);

// ── Legends for the all-categories and group-comparison charts ─────
(function(){
  const legAll = document.getElementById('legend-textile-usage-all');
  DATA.textile_usage_category_labels_all.forEach(label => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<div class="legend-dot" style="background:${TEXTILE_ALL_CATEGORY_COLORS[label]||'#ccc'}"></div>${label}`;
    legAll.appendChild(item);
  });
  const legGroup = document.getElementById('legend-textile-usage-group');
  DATA.textile_usage_group_labels.forEach(label => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<div class="legend-dot" style="background:${TEXTILE_GROUP_COLORS[label]||'#ccc'}"></div>${label}`;
    legGroup.appendChild(item);
  });
})();

// ── All seven textile usage categories (Usage section) ────────────
makeUsageTotal('chart-textile-usage-all-total', DATA.textile_usage_counts_all, TEXTILE_ALL_CATEGORY_COLORS);
makeUsageYear('chart-textile-usage-all-year', DATA.year_textile_usage_all, DATA.textile_usage_category_labels_all,
              TEXTILE_ALL_CATEGORY_COLORS, DATA.all_years);

// ── Metaphorical vs. Non-metaphorical group totals (Overview) ──────
makeUsageTotal('chart-textile-usage-group-total', DATA.textile_usage_group_counts, TEXTILE_GROUP_COLORS);

// ── Textile Metaphor per word — stacked horizontal bar ────────────
(function(){
  const wordUsage = DATA.word_textile_usage;
  if (!wordUsage || Object.keys(wordUsage).length === 0) return;
  const catLabels = DATA.textile_usage_category_labels;

  // Sort words by Textile Metaphor count descending
  const focusCat = catLabels[0] || '';
  const words = Object.keys(wordUsage).sort((a, b) => {
    const aVal = (wordUsage[a].find(([l])=>l===focusCat)||[,0])[1];
    const bVal = (wordUsage[b].find(([l])=>l===focusCat)||[,0])[1];
    return bVal - aVal;
  });

  // Convert per-word usage arrays to a lookup map
  const usageLookup = {};
  words.forEach(w => {
    usageLookup[w] = {};
    wordUsage[w].forEach(([l,n]) => { usageLookup[w][l] = n; });
  });

  new Chart(document.getElementById('chart-textile-usage-per-word'), {
    type: 'bar',
    data: {
      labels: words,
      datasets: catLabels.map(label => ({
        label,
        data: words.map(w => (usageLookup[w]||{})[label] || 0),
        backgroundColor: TEXTILE_USAGE_COLORS[label] || '#c8c8c8',
        stack: 'a',
        borderWidth: 0,
      }))
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { position:'right', labels:{ boxWidth:10, padding:8, font:{ size:10 } } } },
      scales: {
        x: { stacked: true, grid: GRID, ticks: TICKS, beginAtZero: true },
        y: { stacked: true, grid: { display:false }, ticks: { ...TICKS, font:{ family:"'JetBrains Mono',monospace", size:11 } } }
      }
    }
  });
})();
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# 5.  Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CSV_PATH.exists():
        sys.exit(f"ERROR: CSV not found: {CSV_PATH}")
    if not CLEAN_CSV_PATH.exists():
        sys.exit(f"ERROR: Clean CSV not found: {CLEAN_CSV_PATH}")

    print(f"Loading {CSV_PATH} …")
    loaded_rows = load_csv(CSV_PATH)
    all_rows    = drop_doublettes(loaded_rows)
    rows        = filter_rows(all_rows)
    print(f"  {len(loaded_rows)} rows total, {len(loaded_rows) - len(all_rows)} doublette rows dropped, "
          f"{len(rows)} included (include_exclude = y).")

    print(f"Loading {CLEAN_CSV_PATH} …")
    clean_rows = load_csv(CLEAN_CSV_PATH)
    print(f"  {len(clean_rows)} textile occurrence rows "
          f"(used for textile co-occurrence/collocation only).")

    print("Computing statistics …")
    stats = build_stats(rows, all_rows, clean_rows)
    print(f"  Done. {len(stats['textile_words'])} textile words, "
          f"{stats['construction_texts']} construction texts.")

    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json.dumps(stats, ensure_ascii=False))
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"\nDashboard written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()