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
  * Journal-counts table — total articles published per DH journal per year,
    independent of this corpus. Used only to contextualise the corpus against
    overall publication volume (see build_stats vs. compute_survey_coverage).

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
# One row per textile-word occurrence, pre-filtered to include_exclude == "y".
# Used only for the co-occurrence and collocation statistics; every other
# statistic is computed from CSV_PATH.
CLEAN_CSV_PATH = Path("../CLEAN Weaving DH Data Table.csv")
# Total articles published per DH journal per year, independent of this
# corpus's search hits. Feeds compute_survey_coverage() only.
JOURNAL_COUNTS_PATH = Path("../journal_count/articles_per_year_long.csv")
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

# Only Textile Metaphor is a recognised category; General Metaphor and every
# other annotated category are excluded from all counts.
TEXTILE_USAGE_CATEGORIES = {"textile metaphor"}

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

# ---------------------------------------------------------------------------
# 3.  Aggregate statistics
# ---------------------------------------------------------------------------

def _accumulate_textile_cooc_from_clean(
        clean_rows: list[dict],
        cooc_acc: defaultdict, colloc_acc: defaultdict,
        year_cooc_acc: defaultdict, year_colloc_acc: defaultdict) -> None:
    """Populate the co-occurrence/collocation accumulators from the clean,
    one-row-per-occurrence table (CLEAN_CSV_PATH)."""
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
    """
    Compute every dashboard statistic from `rows` (included FULL-table texts),
    restricted throughout to texts with a Textile Metaphor use. `clean_rows`
    drives the co-occurrence/collocation stats only and falls back to `rows`
    if not given.
    """
    years         : list[int]   = []
    sources       : Counter     = Counter()
    source_years  : defaultdict = defaultdict(list)
    textile_freq  : Counter     = Counter()
    year_textile  : defaultdict = defaultdict(Counter)

    year_text_counts: Counter = Counter()  # Textile Metaphor texts per year, used to normalise hit rates

    textile_cooc_acc  : defaultdict = defaultdict(Counter)
    textile_colloc_acc: defaultdict = defaultdict(Counter)
    year_cooc_textile  : defaultdict = defaultdict(lambda: defaultdict(Counter))
    year_colloc_textile: defaultdict = defaultdict(lambda: defaultdict(Counter))

    source_textile_by_cat : defaultdict = defaultdict(lambda: defaultdict(int))

    textile_metaphor_texts   : int = 0

    # Distinct comparison vocabulary, not a Textile Metaphor category —
    # logged to the console only, not displayed in the dashboard.
    construction_texts    : int = 0

    for row in rows:
        c_words = parse_words(row.get("construction_words", ""))
        if c_words:
            construction_texts += 1

        t_usage_raw = _get_column_by_substr(row, "usage_textile")
        t_cats = parse_usage_categories(t_usage_raw, TEXTILE_USAGE_VARIANTS, TEXTILE_USAGE_CATEGORIES)
        if not t_cats:
            continue  # no Textile Metaphor use: excluded from every statistic below

        source = row.get("journal_title", "").strip()
        year   = _parse_year(row)
        t_words = parse_words(row.get("textile_words", ""))

        textile_metaphor_texts += 1
        if year:
            years.append(year)
            year_text_counts[year] += 1
        if source:
            sources[source] += 1
            if year:
                source_years[source].append(year)

        for w in t_words:
            w = textile_canonical(w)
            textile_freq[w] += 1
            if year:
                year_textile[year][w] += 1

        for cat in t_cats:
            if source:
                source_textile_by_cat[source][TEXTILE_USAGE_LABELS[cat]] += 1

    _accumulate_textile_cooc_from_clean(
        clean_rows if clean_rows is not None else rows,
        textile_cooc_acc, textile_colloc_acc, year_cooc_textile, year_colloc_textile)

    year_range  = (min(years), max(years)) if years else (None, None)
    year_counts : Counter = Counter(years)
    all_years   = list(range(year_range[0], year_range[1] + 1)) if year_range[0] else []

    # Normalise temporal co-occurrence/collocation by Textile Metaphor text count per year
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
        "textile_freq"     : textile_freq.most_common(),
        "year_textile"     : {y: dict(c) for y, c in year_textile.items()},
        "textile_cooc"     : top_cooc(textile_cooc_acc),
        "textile_colloc"   : top_colloc(textile_colloc_acc),
        "year_cooc_textile"  : normalise_year_cooc(year_cooc_textile),
        "year_colloc_textile": normalise_year_cooc(year_colloc_textile),
        "textile_words"    : [w for w, _ in textile_freq.most_common()],
        "textile_variants" : TEXTILE_VARIANTS,
        "source_textile_by_cat"  : {
            s: dict(cats) for s, cats in source_textile_by_cat.items()
        },
        "construction_texts"     : construction_texts,
    }

# ---------------------------------------------------------------------------
# 4.  Survey-coverage statistics
# ---------------------------------------------------------------------------
#
# The FULL table holds one row per search hit, not every article a journal
# ever published, so it can't say how common Textile Metaphor use is
# relative to total output. JOURNAL_COUNTS_PATH supplies that denominator
# for 7 of this corpus's 8 sources ("Digital Medievalist" has no counterpart
# there and is excluded here, as is any journal-year absent from the survey).

# Maps journal_title values from the FULL table to the journal names used in
# ../journal_count. Renamed/merged journals map to their combined entry.
JOURNAL_COUNTS_NAME_MAP: dict[str, str] = {
    "Index of DH Conferences": "Index of DH Conferences",
    "Digital Humanities Quarterly": "Digital Humanities Quarterly",
    "Literary and Linguistic Computing/Digital Scholarship in the Humanities":
        "Literary and Linguistic Computing / Digital Scholarship in the Humanities",
    "Computers and Translation/Computers and the Humanities/Machine Translation/Language Resources and Evaluation":
        "Computers and the Humanities / Language Resources and Evaluation",
    "Journal of Cultural Analytics": "Journal of Cultural Analytics",
    "Journal of the Text Encoding Initiative": "Journal of the Text Encoding Initiative",
    "Digital Classics Online": "Digital Classics Online",
}

def load_journal_counts(path: Path) -> dict[str, dict[int, int]]:
    """Load ../journal_count's {journal;year;count} survey into
    {journal_name: {year: total_articles_published}}."""
    counts: defaultdict = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            counts[row["journal"]][int(row["year"])] = int(row["count"])
    return counts

def compute_survey_coverage(all_rows: list[dict], rows: list[dict],
                             journal_counts: dict[str, dict[int, int]],
                             all_years: list[int]) -> dict:
    """
    Context statistics for the journal-years this corpus actually searched
    (from all_rows — every search-hit row, included or excluded, since an
    excluded row still proves that journal-year was searched) and for
    sources with a counterpart in journal_counts:
      * articles_surveyed_total   — total articles those journals actually
        published in those years (the search population).
      * textile_metaphor_rate_pct — Textile Metaphor texts among the matched
        journals, as a percentage of articles_surveyed_total.
      * survey_by_year   — {year: total articles}, aligned with all_years.
      * survey_by_source — {journal_title: total articles}, keyed like
        DATA.sources for the per-source chart.
    """
    searched_years_by_mapped: defaultdict = defaultdict(set)
    searched_years_by_source: defaultdict = defaultdict(set)
    for row in all_rows:
        title  = row.get("journal_title", "").strip()
        mapped = JOURNAL_COUNTS_NAME_MAP.get(title)
        year   = _parse_year(row)
        if mapped and year:
            searched_years_by_mapped[mapped].add(year)
            searched_years_by_source[title].add(year)

    articles_surveyed_total = sum(
        journal_counts.get(journal, {}).get(year, 0)
        for journal, years in searched_years_by_mapped.items()
        for year in years
    )

    textile_metaphor_matched = 0
    for row in rows:
        if row.get("journal_title", "").strip() not in JOURNAL_COUNTS_NAME_MAP:
            continue
        t_usage_raw = _get_column_by_substr(row, "usage_textile")
        t_cats = parse_usage_categories(t_usage_raw, TEXTILE_USAGE_VARIANTS, TEXTILE_USAGE_CATEGORIES)
        if t_cats:
            textile_metaphor_matched += 1

    rate = (round(textile_metaphor_matched / articles_surveyed_total * 100, 2)
            if articles_surveyed_total else None)

    survey_by_year_raw: Counter = Counter()
    for journal, years in searched_years_by_mapped.items():
        for year in years:
            survey_by_year_raw[year] += journal_counts.get(journal, {}).get(year, 0)
    survey_by_year = {y: survey_by_year_raw.get(y, 0) for y in all_years}

    survey_by_source = {}
    for title, years in searched_years_by_source.items():
        mapped = JOURNAL_COUNTS_NAME_MAP.get(title)
        survey_by_source[title] = sum(journal_counts.get(mapped, {}).get(y, 0) for y in years)

    return {
        "articles_surveyed_total"   : articles_surveyed_total,
        "textile_metaphor_rate_pct" : rate,
        "survey_by_year"            : survey_by_year,
        "survey_by_source"          : survey_by_source,
    }

# ---------------------------------------------------------------------------
# 5.  HTML template
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
  .source-table th:last-child, .source-table td:last-child { text-align: right; padding-right: 0; }
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
    <div class="stat-card"><div class="stat-number" id="stat-articles-surveyed">—</div><div class="stat-label">Articles surveyed</div></div>
    <div class="stat-card"><div class="stat-number" id="stat-textile-rate">—</div><div class="stat-label">Textile metaphor rate</div></div>
  </div>
  <br>
  <p class="chart-title">Sources in corpus</p>
  <table class="source-table">
    <thead><tr><th>Source</th><th>Texts with Textile Metaphors</th></tr></thead>
    <tbody id="source-tbody"></tbody>
  </table>
</section>

<section id="temporal">
  <p class="section-label">Chronology</p>
  <h2 class="section-title">Distribution Over Time</h2>
  <div class="chart-row">
    <div class="chart-wrap">
      <p class="chart-title">Texts with textile metaphors per year</p>
      <canvas id="chart-year-total" height="140"></canvas>
      <p class="chart-note">Distinct texts per year with a Textile Metaphor use.</p>
    </div>
    <div class="chart-wrap">
      <p class="chart-title">All articles per year</p>
      <canvas id="chart-year-all" height="140"></canvas>
      <p class="chart-note">Total articles published per year.</p>
    </div>
  </div>

  <p class="subsection-title">Textile word hits per year</p>
  <div class="chart-wrap">
    <p class="chart-title">By word</p>
    <canvas id="chart-year-textile" height="160"></canvas>
    <p class="chart-note">Absolute hits per year, by textile word.</p>
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
      <p class="chart-note">Absolute hits per year for the selected word.</p>
    </div>
  </div>
</section>

<section id="sources">
  <p class="section-label">Sources</p>
  <h2 class="section-title">By Source</h2>
  <div class="chart-row">
    <div class="chart-wrap">
      <p class="chart-title">Textile Metaphor Texts per Source</p>
      <canvas id="chart-source-ratio" height="160"></canvas>
      <p class="chart-note">Texts per source in which a Textile Metaphor word appeared.</p>
    </div>
    <div class="chart-wrap">
      <p class="chart-title">All Articles per Source</p>
      <canvas id="chart-source-all" height="160"></canvas>
      <p class="chart-note">Total articles published.</p>
    </div>
  </div>
</section>

<section id="cooccurrence">
  <p class="section-label">Context</p>
  <h2 class="section-title">Co-occurrence &amp; Collocation</h2>

  <p class="subsection-title">Top 5 Co-occurring Words in KWIC Context</p>
  <p style="font-size:0.85rem;color:var(--grey-2);margin-bottom:1rem;">Most frequent lemmatised content words within a ±15-token KWIC window, drawn from Textile Metaphor occurrences only. Co-occurrence analysis uses WordNet noun lemmatization, which differs from the PorterStemmer used at KWIC search time.</p>
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

document.getElementById('stat-textile-metaphor').textContent = DATA.textile_metaphor_texts.toLocaleString();

document.getElementById('stat-articles-surveyed').textContent =
  DATA.articles_surveyed_total != null ? DATA.articles_surveyed_total.toLocaleString() : '—';
document.getElementById('stat-textile-rate').textContent =
  DATA.textile_metaphor_rate_pct != null ? `${String(DATA.textile_metaphor_rate_pct).replace('.', ',')}%` : '—';

const tbody = document.getElementById('source-tbody');
DATA.sources.forEach(([j,n]) => {
  const yr    = DATA.source_years[j];
  const yrStr = yr ? (yr[0]===yr[1] ? ` (${yr[0]})` : ` (${yr[0]}–${yr[1]})`) : '';
  const tr    = document.createElement('tr');
  tr.innerHTML = `<td>${j}${yrStr}</td><td>${n}</td>`;
  tbody.appendChild(tr);
});

const textileVariantParts = Object.entries(DATA.textile_variants).map(([w,v])=>`${w} (incl. ${v})`);
document.getElementById('textile-variants-note').textContent =
  'Morphological variants included: ' + textileVariantParts.join('; ') + '.';

// ── Source chart ──────────────────────────────────────────────────
// Two separate charts, each on its own y-axis, rather than one overlay: the
// Textile Metaphor rate is small enough (~1-2%) that a dark-blue bar drawn
// over a light-blue one at true scale would be all but invisible.
(function(){
  const sources = DATA.sources.map(([s])=>s);

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

  new Chart(document.getElementById('chart-source-all'), {
    type: 'bar',
    data: {
      labels: sources,
      datasets: [{
        data: sources.map(s => DATA.survey_by_source[s] || 0),
        backgroundColor: '#9dc3e6',
        borderWidth: 0,
      }]
    },
    options: {
      plugins:{ legend:{ display:false } },
      scales:{
        x:{ grid:GRID, ticks:{ ...TICKS, maxRotation:30, font:{size:10} } },
        y:{ grid:GRID, ticks:TICKS, beginAtZero:true }
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

new Chart(document.getElementById('chart-year-all'), {
  type: 'bar',
  data: {
    labels: DATA.all_years,
    datasets: [{ data: DATA.all_years.map(y=>DATA.survey_by_year[y]||0), backgroundColor:'#c8c8c8', borderWidth:0 }]
  },
  options: { plugins:{legend:{display:false}}, scales:{ x:{grid:GRID,ticks:TICKS}, y:{grid:GRID,ticks:TICKS,beginAtZero:true} } }
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
# 6.  Main
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

    if JOURNAL_COUNTS_PATH.exists():
        print(f"Loading {JOURNAL_COUNTS_PATH} …")
        journal_counts = load_journal_counts(JOURNAL_COUNTS_PATH)
        stats.update(compute_survey_coverage(all_rows, rows, journal_counts, stats["all_years"]))
        print(f"  Articles surveyed: {stats['articles_surveyed_total']}, "
              f"Textile Metaphor rate: {stats['textile_metaphor_rate_pct']}%.")
    else:
        print(f"  NOTE: {JOURNAL_COUNTS_PATH} not found — skipping survey-coverage statistics.")
        stats["articles_surveyed_total"]   = None
        stats["textile_metaphor_rate_pct"] = None
        stats["survey_by_year"]            = {}
        stats["survey_by_source"]          = {}

    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json.dumps(stats, ensure_ascii=False))
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"\nDashboard written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()