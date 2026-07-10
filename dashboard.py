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
  * Textile Metaphor is the only recognised usage category — General Metaphor
    and every other annotated category are excluded from every count in the
    dashboard. A text is counted once per category, however often its words
    recur, and only among included texts.
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

# Canonical forms for the textile metaphor usage categories (case-insensitive).
# General Metaphor (and every other annotated category) is intentionally
# excluded — it is not counted anywhere in the dashboard; only Textile
# Metaphor is treated as a recognised category.
TEXTILE_USAGE_CATEGORIES  = {"textile metaphor"}

# Canonical display labels (title-case) for each normalised key
TEXTILE_USAGE_LABELS: dict[str, str] = {
    "textile metaphor": "Textile Metaphor",
}

# Surface-form variants that normalise to a canonical category key.
TEXTILE_USAGE_VARIANTS: dict[str, str] = {
    "textile metaphor": "textile metaphor",
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
        year_cooc_acc: defaultdict, year_colloc_acc: defaultdict) -> None:
    """
    Populate the textile co-occurrence / collocation accumulators from the
    clean, one-row-per-occurrence table (CLEAN_CSV_PATH). Each row already
    represents exactly one Textile Metaphor word hit with its own concordance
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

        cooc_counts   = _cooc_counter(kwic)
        colloc_counts = _colloc_counter(kwic)

        for lemma, cnt in cooc_counts.items():
            cooc_acc[canon][lemma] += cnt
            if year is not None:
                year_cooc_acc[year][canon][lemma] += cnt
        for lemma, cnt in colloc_counts.items():
            colloc_acc[canon][lemma] += cnt
            if year is not None:
                year_colloc_acc[year][canon][lemma] += cnt

def build_stats(rows: list[dict], clean_rows: Optional[list[dict]] = None) -> dict:
    # rows        – texts with include_exclude == "y"; every statistic below is
    #               computed from this set (one row per text in the FULL table)
    # clean_rows  – rows from the clean, one-occurrence-per-row textile KWIC
    #               table (CLEAN_CSV_PATH); drives the textile co-occurrence
    #               and collocation/concordance stats. Falls back to `rows`
    #               (the FULL table, with its multi-word KWIC blocks) if not
    #               given, so standalone calls to build_stats keep working.
    years         : list[int]   = []
    sources       : Counter     = Counter()
    source_years  : defaultdict = defaultdict(list)
    textile_freq  : Counter     = Counter()
    year_textile  : defaultdict = defaultdict(Counter)

    year_text_counts: Counter = Counter()  # texts per year, used to normalise hit rates

    textile_cooc_acc  : defaultdict = defaultdict(Counter)
    textile_colloc_acc: defaultdict = defaultdict(Counter)
    year_cooc_textile  : defaultdict = defaultdict(lambda: defaultdict(Counter))
    year_colloc_textile: defaultdict = defaultdict(lambda: defaultdict(Counter))

    source_textile_by_cat : defaultdict = defaultdict(lambda: defaultdict(int))

    # Distinct texts with a Textile Metaphor use (headline stat). General
    # Metaphor and every other annotated category are not counted here.
    textile_metaphor_texts   : int = 0

    # Construction is a plain presence flag: one entry per text that contains a
    # construction word, with no sub-categories.
    construction_texts    : int = 0

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

        # Parse the textile usage-category cell (may hold several comma-separated
        # values); only Textile Metaphor is a recognised category.
        t_usage_raw = _get_column_by_substr(row, "usage_textile")
        t_cats = parse_usage_categories(t_usage_raw, TEXTILE_USAGE_VARIANTS, TEXTILE_USAGE_CATEGORIES)

        # Word-level counts (Frequency / Distribution Over Time / By Source)
        # only draw on texts carrying a Textile Metaphor use; a text with no
        # such use contributes nothing here, however many textile words it
        # otherwise mentions (e.g. as Textile Reference or Verb).
        if t_cats:
            textile_metaphor_texts += 1
            for w in t_words:
                w = textile_canonical(w)
                textile_freq[w] += 1
                if year:
                    year_textile[year][w] += 1
        for cat in t_cats:
            if source:
                source_textile_by_cat[source][TEXTILE_USAGE_LABELS[cat]] += 1

        # Total construction texts, once per text if it contains a construction word
        if c_words:
            construction_texts += 1

    # Textile co-occurrence / collocation (Kookurrenz / Konkordanz) come from
    # the clean, one-row-per-occurrence table rather than `rows` (FULL); that
    # table holds Textile Metaphor occurrences only.
    _accumulate_textile_cooc_from_clean(
        clean_rows if clean_rows is not None else rows,
        textile_cooc_acc, textile_colloc_acc, year_cooc_textile, year_colloc_textile)

    year_range  = (min(years), max(years)) if years else (None, None)
    year_counts : Counter = Counter(years)
    all_years   = list(range(year_range[0], year_range[1] + 1)) if year_range[0] else []

    # Normalise temporal co-occurrence/collocation by text count per year
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
        "all_years"        : all_years,
        "year_counts"      : {y: year_counts.get(y, 0) for y in all_years},
        "year_text_counts" : {y: year_text_counts.get(y, 0) for y in all_years},
        # Textile Metaphor word frequency, corpus-wide
        "textile_freq"     : textile_freq.most_common(),
        # Absolute (not normalised) Textile Metaphor word hits per year
        "year_textile"     : {y: dict(c) for y, c in year_textile.items()},
        "textile_cooc"     : top_cooc(textile_cooc_acc),
        "textile_colloc"   : top_colloc(textile_colloc_acc),
        "year_cooc_textile"  : normalise_year_cooc(year_cooc_textile),
        "year_colloc_textile": normalise_year_cooc(year_colloc_textile),
        "textile_words"    : [w for w, _ in textile_freq.most_common()],
        "textile_variants" : TEXTILE_VARIANTS,
        # Textile Metaphor texts per source: {source -> {display_label -> count}}
        "source_textile_by_cat"  : {
            s: dict(cats) for s, cats in source_textile_by_cat.items()
        },
        # Total construction texts corpus-wide, text-level
        "construction_texts"     : construction_texts,
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
    <li><a href="#vocabulary">Words</a></li>
    <li><a href="#sources">Sources</a></li>
    <li><a href="#cooccurrence">Context</a></li>
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
  </div>
  <br>
  <p class="chart-title">Sources in corpus</p>
  <table class="source-table">
    <thead><tr><th>Source</th><th>Texts</th></tr></thead>
    <tbody id="source-tbody"></tbody>
  </table>
</section>

<section id="temporal">
  <p class="section-label">Chronology</p>
  <h2 class="section-title">Distribution Over Time</h2>
  <p style="font-size:0.85rem;color:var(--grey-2);margin-bottom:1.5rem;">
    How the corpus and its Textile Metaphor use spread across publication years. The first
    chart is the raw number of included texts per year, regardless of textile-word use; the
    second shows, per year, the absolute number of hits per textile word — counting only
    words from texts with a Textile Metaphor use.
  </p>
  <div class="chart-wrap">
    <p class="chart-title">Texts per year</p>
    <canvas id="chart-year-total" height="80"></canvas>
    <p class="chart-note">Number of included texts published each year (all included texts, not restricted to Textile Metaphor).</p>
  </div>

  <p class="subsection-title">Textile word hits per year</p>
  <div class="chart-wrap">
    <p class="chart-title">By word</p>
    <canvas id="chart-year-textile" height="160"></canvas>
    <p class="chart-note">Absolute hits per year, by textile word — Textile Metaphor texts only.</p>
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
      <p class="chart-note">Textile Metaphor texts only. Search used unlemmatised surface forms; morphological variants (shown above) were grouped manually in post-processing.</p>
    </div>
    <div class="chart-wrap">
      <p class="chart-title">Temporal distribution — select word</p>
      <div class="tab-group" id="tabs-textile"></div>
      <canvas id="chart-textile-word-time" height="160"></canvas>
      <p class="chart-note">Absolute hits per year for the selected word — Textile Metaphor texts only.</p>
    </div>
  </div>
</section>

<section id="sources">
  <p class="section-label">Sources</p>
  <h2 class="section-title">By Source</h2>
  <p class="subsection-title">Textile Metaphor Texts per Source</p>
  <div class="chart-wrap">
    <canvas id="chart-source-ratio" height="120"></canvas>
    <p class="chart-note">Texts per source in which a Textile Metaphor word appeared — Textile Metaphor texts only.</p>
  </div>
</section>

<section id="cooccurrence">
  <p class="section-label">Context</p>
  <h2 class="section-title">Co-occurrence &amp; Collocation</h2>

  <p class="subsection-title">Top 5 Co-occurring Words in KWIC Context</p>
  <p style="font-size:0.85rem;color:var(--grey-2);margin-bottom:1rem;">Most frequent lemmatised content words within the ±15-token KWIC window, drawn from Textile Metaphor occurrences only. Co-occurrence analysis uses WordNet noun lemmatization, which differs from the PorterStemmer used at KWIC search time.</p>
  <div class="cooc-grid" id="cooc-textile"></div>

  <p class="subsection-title">Immediate Collocations (±2 tokens)</p>
  <p style="font-size:0.85rem;color:var(--grey-2);margin-bottom:1rem;">Words appearing directly adjacent to the hit token, revealing typical phrasal patterns (e.g. <em>building a corpus</em>, <em>weaving together</em>), drawn from Textile Metaphor occurrences only. Left and right positions are pooled.</p>
  <div class="cooc-grid" id="colloc-textile"></div>

  <p class="subsection-title">Temporal Co-occurrence &amp; Collocation Trend</p>
  <p style="font-size:0.85rem;color:var(--grey-2);margin-bottom:1.5rem;">Frequency of a selected co-occurring or collocating term alongside a given Textile Metaphor word, normalised per text per year.</p>
  <div class="chart-row">
    <div class="chart-wrap">
      <p class="chart-title">Co-occurrence — select word, then co-occurring term</p>
      <div class="tab-group" id="tabs-trend-textile-word"></div>
      <div class="tab-group" id="tabs-trend-textile-lemma"></div>
      <canvas id="chart-trend-textile" height="160"></canvas>
    </div>
    <div class="chart-wrap">
      <p class="chart-title">Collocation — select word, then collocate</p>
      <div class="tab-group" id="tabs-trend-textile-colloc-word"></div>
      <div class="tab-group" id="tabs-trend-textile-colloc-lemma"></div>
      <canvas id="chart-trend-textile-colloc" height="160"></canvas>
    </div>
  </div>
</section>

<section id="methodology">
  <h2 class="section-title">Notes</h2>
  <p class="chart-note">
    Corpus comprises all journal articles in which at least one textile metaphor word was identified and manually confirmed as metaphorical. Texts were fully extracted including abstracts; OCR quality was not manually verified and fuzzy OCR artefacts may affect word counts. Include/exclude decisions were made by a single annotator. Search used unlemmatised surface forms; morphological variants were added manually where relevant.
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

// ── Overview ──────────────────────────────────────────────────────
document.getElementById('stat-sources').textContent = DATA.sources.length;
document.getElementById('stat-years').textContent   =
  DATA.year_range[0] ? `${DATA.year_range[0]}–${DATA.year_range[1]}` : '—';

// Textile Metaphor texts: rows tagged Textile Metaphor, counted once per text
// (computed in Python, not re-derived here)
document.getElementById('stat-textile-metaphor').textContent = DATA.textile_metaphor_texts.toLocaleString();

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

// ── Source chart ──────────────────────────────────────────────────
(function(){
  const sources = DATA.sources.map(([s])=>s);

  // Textile Metaphor texts per source
  new Chart(document.getElementById('chart-source-ratio'), {
    type: 'bar',
    data: {
      labels: sources,
      datasets: [{
        data: sources.map(s => (DATA.source_textile_by_cat[s]||{})['Textile Metaphor'] || 0),
        backgroundColor: '#2563a8',
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

// ── Stacked temporal by word ───────────────────────────────────────
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

// ── Per-word temporal with tabs ────────────────────────────────────
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

// ── Temporal co-occurrence/collocation trend (normalised) ──────────
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
makeTrendChart('chart-trend-textile-colloc','tabs-trend-textile-colloc-word','tabs-trend-textile-colloc-lemma',
               DATA.year_colloc_textile, DATA.all_years);

// ── Co-occurrence / collocation cards ──────────────────────────────
renderCooc('cooc-textile', DATA.textile_cooc, DATA.textile_variants);
renderCooc('colloc-textile', DATA.textile_colloc, DATA.textile_variants);
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
    stats = build_stats(rows, clean_rows)
    print(f"  Done. {len(stats['textile_words'])} textile words, "
          f"{stats['construction_texts']} construction texts.")

    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json.dumps(stats, ensure_ascii=False))
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"\nDashboard written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()