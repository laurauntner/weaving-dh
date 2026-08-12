"""
Build a self-contained, interactive HTML dashboard on metaphorical textile
vocabulary in a corpus of digital-humanities journal articles.

Input (CSV, see Configuration below):
  * CLEAN table — one row per confirmed Textile Metaphor word occurrence,
    already restricted to the texts and words a human reviewer confirmed.
    The sole data source for every statistic in the dashboard.
  * Journal-counts table — total articles published per DH journal per year,
    independent of this corpus. Used only to contextualise the corpus against
    overall publication volume (see build_stats vs. compute_survey_coverage).

Output:
  * A single HTML file (OUTPUT_PATH) with the computed statistics embedded as
    JSON and rendered client-side with Chart.js.
  * A standalone CSV (STATISTICS_CSV_PATH) with every statistic shown in the
    dashboard, in tidy long format, for sharing without KWIC/concordance text.
  * A folder of PNGs (IMAGES_DIR) — every chart in the dashboard, including
    every tab of "select word"/"select source" charts, rendered via a
    headless browser (see export_chart_images; requires Playwright).

Counting conventions:
  * Every row in CLEAN_CSV_PATH is a confirmed Textile Metaphor occurrence —
    there is no other category to filter by and no include/exclude flag.
  * A text is counted once per statistic (once per word it uses, once per
    source, once per year), however many occurrence rows it has; only the
    co-occurrence/collocation statistics count at occurrence-level instead.
"""

import base64
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

# Holds raw KWIC excerpts from copyrighted articles, so it lives outside
# this repo (not publishable) rather than under data/. The sole data source
# for every statistic in the dashboard.
CLEAN_CSV_PATH = Path("../../CLEAN Weaving DH Data Table.csv")
# Total articles published per DH journal per year, independent of this
# corpus's search hits. Feeds compute_survey_coverage() only.
JOURNAL_COUNTS_PATH = Path("../data/articles_per_year_long.csv")
OUTPUT_PATH = Path("../html/textile_metaphors_in_dh_scholarship.html")

TOP_N_COOC   = 5
TOP_N_COLLOC = 8
COLLOC_WIN   = 2

STOPWORDS = set(stopwords.words("english")) | {
    "also", "would", "could", "one", "two", "three", "may", "use",
    "used", "using", "well", "within", "across", "however", "thus",
    "therefore", "whether", "though", "even", "much", "many", "first",
    "second", "new", "based", "see", "et", "al", "pp", "fig",
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

_lemmatizer = WordNetLemmatizer()

def lemmatize(word: str) -> str:
    return _lemmatizer.lemmatize(word.lower(), pos="n")

_WORD_RE  = re.compile(r"\b[a-zA-Z]{3,}\b")
_HIT_RE   = re.compile(r"\*\*(\w+)\*\*")
_LABEL_RE = re.compile(r"^\w[\w\s]*:$")

# ---------------------------------------------------------------------------
# 1.  Load CSV
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    """Load a CSV/semicolon-CSV, auto-detecting the delimiter — CLEAN_CSV_PATH
    gets re-saved from spreadsheet software from time to time, which can
    flip it between comma- and semicolon-delimited."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        except csv.Error:
            dialect = csv.excel
        return list(csv.DictReader(fh, dialect=dialect))

def _parse_year(row: dict) -> Optional[int]:
    year_raw = row.get("pub_year", "").strip()
    try:
        return int(float(year_raw)) if year_raw else None
    except ValueError:
        return None

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

def build_stats(clean_rows: list[dict]) -> dict:
    """
    Compute every dashboard statistic from clean_rows (CLEAN_CSV_PATH): one
    row per confirmed Textile Metaphor word occurrence. Occurrence rows are
    grouped by text id first, so a text is counted once per statistic
    (once per word it uses, once per source, once per year) regardless of
    how many occurrence rows it has; only co-occurrence/collocation count
    at occurrence-level, directly from clean_rows.
    """
    texts: dict[str, dict] = {}
    text_words: defaultdict = defaultdict(set)
    for row in clean_rows:
        tid = row.get("id", "").strip()
        if not tid:
            continue
        if tid not in texts:
            texts[tid] = {
                "year"  : _parse_year(row),
                "source": row.get("journal_title", "").strip(),
            }
        word = row.get("textile_words", "").strip()
        if word:
            text_words[tid].add(textile_canonical(word))

    years         : list[int]   = []
    sources       : Counter     = Counter()
    source_years  : defaultdict = defaultdict(list)
    textile_freq  : Counter     = Counter()
    year_textile  : defaultdict = defaultdict(Counter)

    year_text_counts: Counter = Counter()  # Textile Metaphor texts per year, used to normalise hit rates

    source_textile_by_cat : defaultdict = defaultdict(lambda: defaultdict(int))
    source_year_counts    : defaultdict = defaultdict(Counter)

    for tid, info in texts.items():
        year, source = info["year"], info["source"]
        if year:
            years.append(year)
            year_text_counts[year] += 1
        if source:
            sources[source] += 1
            if year:
                source_years[source].append(year)
                source_year_counts[source][year] += 1
            source_textile_by_cat[source]["Textile Metaphor"] += 1

        for w in text_words[tid]:
            textile_freq[w] += 1
            if year:
                year_textile[year][w] += 1

    textile_cooc_acc  : defaultdict = defaultdict(Counter)
    textile_colloc_acc: defaultdict = defaultdict(Counter)
    year_cooc_textile  : defaultdict = defaultdict(lambda: defaultdict(Counter))
    year_colloc_textile: defaultdict = defaultdict(lambda: defaultdict(Counter))
    _accumulate_textile_cooc_from_clean(
        clean_rows, textile_cooc_acc, textile_colloc_acc, year_cooc_textile, year_colloc_textile)

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
        "textile_metaphor_texts": len(texts),
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
        "source_year_counts"     : {
            s: dict(c) for s, c in source_year_counts.items()
        },
    }

# ---------------------------------------------------------------------------
# 4.  Survey-coverage statistics
# ---------------------------------------------------------------------------
#
# CLEAN_CSV_PATH holds one row per confirmed occurrence, not every article a
# journal ever published, so it can't say how common Textile Metaphor use is
# relative to total output. JOURNAL_COUNTS_PATH supplies that denominator
# for 7 of this corpus's 8 sources ("Digital Medievalist" has no counterpart
# there and is excluded here). The overview figures (articles_surveyed_total,
# textile_metaphor_rate_pct) use each journal's full, unbounded output, since
# the underlying search covered more than all_years; the per-year/per-source
# chart data is bounded to all_years so those charts share one clean axis.

# Maps journal_title values from CLEAN_CSV_PATH to the journal names used in
# JOURNAL_COUNTS_PATH. Renamed/merged journals map to their combined entry.
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
    """Load JOURNAL_COUNTS_PATH's {journal;year;count} survey into
    {journal_name: {year: total_articles_published}}."""
    counts: defaultdict = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            counts[row["journal"]][int(row["year"])] = int(row["count"])
    return counts

def compute_survey_coverage(clean_rows: list[dict],
                             journal_counts: dict[str, dict[int, int]],
                             all_years: list[int]) -> dict:
    """
    Three kinds of figures, all for the 7 of this corpus's 8 sources with a
    counterpart in journal_counts ("Digital Medievalist" has none and is
    excluded):
      * articles_surveyed_total — corpus-wide overview figure, using each
        journal's full, unbounded published output (the actual search
        covered more than just all_years).
      * textile_metaphor_rate_pct — Textile Metaphor texts as a percentage
        of articles published within all_years specifically (the corpus's
        own year range), not the unbounded total above.
      * survey_by_year, survey_by_source, survey_by_source_year — the
        per-year/per-source chart data, likewise bounded to all_years so
        every chart shares a consistent, meaningful axis.
      * full_years, survey_by_year_full — the journal survey's own full year
        range (unbounded by the corpus's year range), for the "All articles
        per year" chart, which should show total publication volume even in
        years before the corpus has any Textile Metaphor texts.
    """
    full_lifetime_total = sum(
        sum(journal_counts.get(mapped, {}).values())
        for mapped in JOURNAL_COUNTS_NAME_MAP.values()
    )
    articles_surveyed_total = full_lifetime_total

    survey_by_year = {
        y: sum(journal_counts.get(mapped, {}).get(y, 0)
               for mapped in JOURNAL_COUNTS_NAME_MAP.values())
        for y in all_years
    }
    articles_surveyed_bounded_total = sum(survey_by_year.values())

    all_survey_years = [
        y for mapped in JOURNAL_COUNTS_NAME_MAP.values()
        for y in journal_counts.get(mapped, {})
    ]
    full_years = (list(range(min(all_survey_years), max(all_survey_years) + 1))
                  if all_survey_years else [])
    survey_by_year_full = {
        y: sum(journal_counts.get(mapped, {}).get(y, 0)
               for mapped in JOURNAL_COUNTS_NAME_MAP.values())
        for y in full_years
    }

    matched_ids: defaultdict = defaultdict(set)
    for row in clean_rows:
        title = row.get("journal_title", "").strip()
        tid   = row.get("id", "").strip()
        if title in JOURNAL_COUNTS_NAME_MAP and tid:
            matched_ids[title].add(tid)
    textile_metaphor_matched = sum(len(ids) for ids in matched_ids.values())

    rate = (round(textile_metaphor_matched / articles_surveyed_bounded_total * 100, 2)
            if articles_surveyed_bounded_total else None)

    survey_by_source: dict = {}
    survey_by_source_year: dict = {}
    for title, mapped in JOURNAL_COUNTS_NAME_MAP.items():
        year_map = {y: journal_counts.get(mapped, {}).get(y, 0) for y in all_years}
        survey_by_source_year[title] = year_map
        survey_by_source[title] = sum(year_map.values())

    return {
        "articles_surveyed_total"   : articles_surveyed_total,
        "textile_metaphor_rate_pct" : rate,
        "survey_by_year"            : survey_by_year,
        "survey_by_source"          : survey_by_source,
        "survey_by_source_year"     : survey_by_source_year,
        "full_years"                : full_years,
        "survey_by_year_full"       : survey_by_year_full,
    }

# ---------------------------------------------------------------------------
# 5.  Statistics CSV export
# ---------------------------------------------------------------------------

STATISTICS_CSV_PATH   = Path("../data/statistics.csv")
STATISTICS_CSV_HEADER = ["metric", "year", "word", "related_word", "source", "value"]

def build_statistics_rows(stats: dict) -> list[list]:
    """
    Flatten every statistic shown in the dashboard into a tidy long-format
    table: one row per (metric, dimensions) with a single `value` column.
    Dimensions that don't apply to a given metric are left blank. Excludes
    all KWIC/concordance text.
    """
    rows: list[list] = []

    def add(metric, value, *, year="", word="", related_word="", source=""):
        rows.append([metric, year, word, related_word, source, value])

    add("sources_count", len(stats["sources"]))
    add("year_range_min", stats["year_range"][0])
    add("year_range_max", stats["year_range"][1])
    add("textile_metaphor_texts_total", stats["textile_metaphor_texts"])
    add("articles_surveyed_total", stats.get("articles_surveyed_total"))
    add("textile_metaphor_rate_pct", stats.get("textile_metaphor_rate_pct"))

    for source, count in stats["sources"]:
        add("source_text_count", count, source=source)
    for source, (y_min, y_max) in stats["source_years"].items():
        add("source_year_min", y_min, source=source)
        add("source_year_max", y_max, source=source)

    survey_by_year = stats.get("survey_by_year", {})
    for year in stats["all_years"]:
        add("year_texts", stats["year_counts"].get(year, 0), year=year)
        add("year_all_articles", survey_by_year.get(year, 0), year=year)

    for year, words in stats["year_textile"].items():
        for word, count in words.items():
            add("year_word_hits", count, year=year, word=word)

    for word, count in stats["textile_freq"]:
        add("word_freq_corpus", count, word=word)

    for word, variants in stats["textile_variants"].items():
        add("word_variants", variants, word=word)

    for source, cats in stats["source_textile_by_cat"].items():
        add("source_text_count_textile_metaphor", cats.get("Textile Metaphor", 0), source=source)

    for source, count in stats.get("survey_by_source", {}).items():
        add("source_all_articles", count, source=source)

    for source, years in stats.get("source_year_counts", {}).items():
        for year, count in years.items():
            add("source_year_texts", count, source=source, year=year)

    for source, years in stats.get("survey_by_source_year", {}).items():
        for year, count in years.items():
            add("source_year_all_articles", count, source=source, year=year)

    for word, pairs in stats["textile_cooc"].items():
        for related_word, count in pairs:
            add("textile_cooccurrence", count, word=word, related_word=related_word)

    for word, pairs in stats["textile_colloc"].items():
        for related_word, count in pairs:
            add("textile_collocation", count, word=word, related_word=related_word)

    for word, lemmas in stats["year_cooc_textile"].items():
        for related_word, year_map in lemmas.items():
            for year, value in year_map.items():
                add("year_cooccurrence_rate", value, word=word, related_word=related_word, year=year)

    for word, lemmas in stats["year_colloc_textile"].items():
        for related_word, year_map in lemmas.items():
            for year, value in year_map.items():
                add("year_collocation_rate", value, word=word, related_word=related_word, year=year)

    return rows

def write_statistics_csv(stats: dict, path: Path) -> None:
    """Write every statistic shown in the dashboard to a standalone CSV, for
    sharing the underlying numbers independent of the HTML output."""
    rows = build_statistics_rows(stats)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(STATISTICS_CSV_HEADER)
        writer.writerows(rows)

# ---------------------------------------------------------------------------
# 6.  HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Textile Metaphors in Digital Humanities Scholarship</title>
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
  .site-title { font-family: var(--font-display); font-size: clamp(1.5rem,3.5vw,2.4rem); letter-spacing: -0.02em; line-height: 1.1; }
  .site-contributors { font-size: 0.85rem; color: var(--grey-3); margin-top: 0.6rem; }
  .header-logo { width: 120px; height: 120px; object-fit: cover; flex-shrink: 0; border: var(--rule); }

  nav { border-bottom: var(--rule); padding: 0.75rem 0; position: sticky; top: 0; background: var(--white); z-index: 100; }
  nav ul { display: flex; align-items: center; gap: 2rem; list-style: none; flex-wrap: wrap; }
  nav a { font-size: 0.8rem; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; color: var(--grey-2); text-decoration: none; }
  nav a:hover { color: var(--black); }
  nav .nav-github { margin-left: auto; }
  nav .nav-github a { display: flex; align-items: center; gap: 0.4rem; }

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
  .stat-year-span { text-transform: none; letter-spacing: normal; color: var(--grey-3); }
  .stat-sub-label { font-size: 0.7rem; color: var(--grey-3); margin-top: 0.2rem; }

  .source-table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
  .source-table th { text-align: left; font-weight: 500; font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--grey-3); padding: 0.5rem 1rem 0.5rem 0; border-bottom: var(--rule); }
  .source-table td { padding: 0.6rem 1rem 0.6rem 0; border-bottom: var(--rule); color: var(--grey-1); }
  .source-table th:last-child, .source-table td:last-child { text-align: right; padding-right: 0; }
  .source-table tr:last-child td { border-bottom: none; }

  .chart-wrap { position: relative; background: #fff; border: var(--rule); padding: 1.5rem; margin-bottom: 1.5rem; }
  .chart-title { font-size: 0.75rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: var(--grey-2); margin-bottom: 1rem; padding-right: 5.5rem; }
  .chart-download { position: absolute; top: 1.4rem; right: 1.5rem; font-size: 0.65rem; font-family: var(--font-mono); letter-spacing: 0.02em; color: var(--grey-3); text-decoration: none; border: 1px solid var(--grey-4); padding: 0.2rem 0.5rem; background: var(--white); cursor: pointer; }
  .chart-download:hover { color: var(--black); border-color: var(--black); }
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
    <div>
      <h1 class="site-title">Textile Metaphors in Digital Humanities Scholarship</h1>
      <p class="site-contributors">Laura Untner, Quinn Daedal, Tessa Gengnagel</p>
    </div>
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
    <li class="nav-github">
      <a href="https://github.com/laurauntner/weaving-dh" target="_blank" rel="noopener noreferrer" aria-label="GitHub repository" title="GitHub repository">
        <svg viewBox="0 0 16 16" width="18" height="18" fill="currentColor" aria-hidden="true"><path d="M8 0c-4.42 0-8 3.58-8 8a8.013 8.013 0 0 0 5.47 7.59c.4.08.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
        <span>GitHub Repository</span>
      </a>
    </li>
  </ul>
</nav>

<section id="overview">
  <p class="section-label">Corpus</p>
  <h2 class="section-title">Overview</h2>
  <div class="stats-grid">
    <div class="stat-card"><div class="stat-number" id="stat-sources">—</div><div class="stat-label">Sources</div><div class="stat-sub-label">(8 surveyed)</div></div>
    <div class="stat-card"><div class="stat-number" id="stat-years">—</div><div class="stat-label">Year range</div><div class="stat-sub-label">(corpus with textile metaphors)</div></div>
    <div class="stat-card"><div class="stat-number" id="stat-textile-metaphor">—</div><div class="stat-label">Metaphorical Textile texts</div></div>
    <div class="stat-card"><div class="stat-number" id="stat-articles-surveyed">—</div><div class="stat-label">Articles surveyed</div></div>
    <div class="stat-card"><div class="stat-number" id="stat-textile-rate">—</div><div class="stat-label">Textile metaphor rate</div><div class="stat-sub-label stat-year-span"></div></div>
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

  <div class="chart-wrap">
    <p class="chart-title">Textile Metaphor rate per year</p>
    <canvas id="chart-year-rate" height="80"></canvas>
    <p class="chart-note">Textile Metaphor texts as a share of all articles published that year. Years without survey data are omitted.</p>
  </div>

  <div class="chart-wrap">
    <p class="chart-title">Textile Metaphor texts vs. all articles (log scale)</p>
    <canvas id="chart-year-log" height="100"></canvas>
    <p class="chart-note">Both series on a shared logarithmic axis, so their growth is directly comparable despite the difference in magnitude. Years with zero Textile Metaphor texts are omitted — undefined on a log scale.</p>
  </div>

  <div class="chart-wrap">
    <p class="chart-title">Indexed growth (<span id="chart-note-baseline-year">—</span> = 100)</p>
    <canvas id="chart-year-indexed" height="100"></canvas>
    <p class="chart-note">Both series rebased to 100 in their first year with data for both, showing relative growth rather than absolute scale.</p>
  </div>

  <p class="subsection-title">Textile word hits per year</p>
  <div class="chart-wrap">
    <p class="chart-title">By word</p>
    <canvas id="chart-year-textile" height="160"></canvas>
    <p class="chart-note">Absolute hits per year, by textile word.</p>
  </div>

  <div class="chart-wrap">
    <p class="chart-title">Textile word hit rate per year, by word</p>
    <canvas id="chart-year-hits-rate" height="200"></canvas>
    <p class="chart-note">Each textile word's hits as a share of all articles published that year. Years without survey data are omitted.</p>
  </div>

  <div class="chart-wrap">
    <p class="chart-title">Textile word hits vs. all articles, by word (log scale)</p>
    <canvas id="chart-year-hits-log" height="200"></canvas>
    <p class="chart-note">Each word plotted against all articles (context, grey) on a shared logarithmic axis, so growth across words of very different frequency is directly comparable. Years with zero hits for a given word are omitted — undefined on a log scale.</p>
  </div>

  <div class="chart-wrap">
    <p class="chart-title">Indexed growth, by word</p>
    <canvas id="chart-year-hits-indexed" height="200"></canvas>
    <p class="chart-note">Each word (and all articles, for context) rebased to 100 in its own first year with a nonzero count, showing relative growth from each word's own starting point rather than absolute scale.</p>
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

  <div class="chart-row">
    <div class="chart-wrap">
      <p class="chart-title">Relative frequency across corpus</p>
      <canvas id="chart-textile-freq-rel"></canvas>
      <p class="chart-note">Each word's hits as a share of Textile Metaphor texts in the corpus.</p>
    </div>
    <div class="chart-wrap">
      <p class="chart-title">Relative temporal distribution — select word</p>
      <div class="tab-group" id="tabs-textile-rel"></div>
      <canvas id="chart-textile-word-time-rel" height="160"></canvas>
      <p class="chart-note">Selected word's hits as a share of all articles published that year. Years without survey data are omitted.</p>
    </div>
  </div>
</section>

<section id="sources">
  <p class="section-label">Sources</p>
  <h2 class="section-title">By Source</h2>
  <div class="chart-wrap">
    <p class="chart-title">Textile Metaphor Texts per Source</p>
    <canvas id="chart-source-ratio" height="120"></canvas>
    <p class="chart-note">Texts per source in which a Textile Metaphor word appeared.</p>
  </div>

  <div class="chart-wrap">
    <p class="chart-title">All Articles per Source <span class="stat-year-span"></span></p>
    <canvas id="chart-source-all" height="120"></canvas>
    <p class="chart-note">Total articles published.</p>
  </div>

  <div class="chart-wrap">
    <p class="chart-title">Textile Metaphor rate by source <span class="stat-year-span"></span></p>
    <canvas id="chart-source-rate" height="120"></canvas>
    <p class="chart-note">Textile Metaphor texts as a share of all articles published, by source. Sources without survey data are omitted.</p>
  </div>

  <div class="chart-wrap">
    <p class="chart-title">Textile Metaphor rate per year, by source</p>
    <canvas id="chart-source-rate-timeline" height="200"></canvas>
    <p class="chart-note">Each source's Textile Metaphor texts as a share of its own articles published that year. Years without survey data for a source are omitted.</p>
  </div>

  <div class="chart-wrap">
    <p class="chart-title">Textile Metaphor texts vs. all articles, by source (log scale)</p>
    <canvas id="chart-source-log" height="200"></canvas>
    <p class="chart-note">Each source's Textile Metaphor text count plotted against total articles (context, grey) on a shared logarithmic axis, so growth across sources of very different output is directly comparable. Years with zero texts for a given source are omitted — undefined on a log scale.</p>
  </div>

  <div class="chart-wrap">
    <p class="chart-title">Indexed growth, by source</p>
    <canvas id="chart-source-indexed" height="200"></canvas>
    <p class="chart-note">Each source (and all articles, for context) rebased to 100 in its own first year with a nonzero count, showing relative growth from each source's own starting point rather than absolute scale.</p>
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

document.getElementById('stat-textile-metaphor').textContent = DATA.textile_metaphor_texts.toLocaleString('en-US');

document.getElementById('stat-articles-surveyed').textContent =
  DATA.articles_surveyed_total != null ? DATA.articles_surveyed_total.toLocaleString('en-US') : '—';
document.getElementById('stat-textile-rate').textContent =
  DATA.textile_metaphor_rate_pct != null ? `${DATA.textile_metaphor_rate_pct}%` : '—';

// "Articles surveyed" / rate-by-source figures are bounded to the corpus's
// own year range, not each journal's full lifetime — noted wherever that
// isn't otherwise visible from a chart's own year axis.
if (DATA.year_range[0]) {
  const yearSpanText = `(${DATA.year_range[0]}–${DATA.year_range[1]})`;
  document.querySelectorAll('.stat-year-span').forEach(el => el.textContent = yearSpanText);
}

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
        x:{ grid:GRID, ticks:{ ...TICKS, autoSkip:false, maxRotation:60, minRotation:60, font:{size:10} } },
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
        x:{ grid:GRID, ticks:{ ...TICKS, autoSkip:false, maxRotation:60, minRotation:60, font:{size:10} } },
        y:{ grid:GRID, ticks:TICKS, beginAtZero:true }
      }
    }
  });

  // Textile Metaphor texts as a percentage of each source's total articles —
  // the relative counterpart to the two absolute-count charts above.
  new Chart(document.getElementById('chart-source-rate'), {
    type: 'bar',
    data: {
      labels: sources,
      datasets: [{
        data: sources.map(s => {
          const total = DATA.survey_by_source[s] || 0;
          const texts = (DATA.source_textile_by_cat[s]||{})['Textile Metaphor'] || 0;
          return total > 0 ? texts / total * 100 : null;
        }),
        backgroundColor: '#2563a8',
        borderWidth: 0,
      }]
    },
    options: {
      plugins:{
        legend:{ display:false },
        tooltip:{ callbacks:{ label: ctx => `${String(Math.round(ctx.parsed.y*10)/10)}%` } },
      },
      scales:{
        x:{ grid:GRID, ticks:{ ...TICKS, maxRotation:30, font:{size:10} } },
        y:{ grid:GRID, ticks:{ ...TICKS, callback: v => `${v}%` }, beginAtZero:true }
      }
    }
  });

  // Same rate, broken out per year and per source — one line per source,
  // colour-matched to the by-word timeline charts above.
  new Chart(document.getElementById('chart-source-rate-timeline'), {
    type: 'line',
    data: {
      labels: DATA.all_years,
      datasets: sources.map((s, i) => ({
        label: s,
        data: DATA.all_years.map(y => {
          const total = (DATA.survey_by_source_year[s]||{})[y] || 0;
          const texts = (DATA.source_year_counts[s]||{})[y] || 0;
          return total > 0 ? texts / total * 100 : null;
        }),
        borderColor: color(i), backgroundColor: color(i),
        spanGaps: false, tension: 0.25, pointRadius: 1.5, borderWidth: 1.5,
      }))
    },
    options: {
      plugins: {
        legend: { position:'right', labels:{boxWidth:10,padding:8,font:{size:10}} },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${String(Math.round(ctx.parsed.y*10)/10)}%` } },
      },
      scales: {
        x: { grid:GRID, ticks:TICKS },
        y: { grid:GRID, ticks:{ ...TICKS, callback: v => `${v}%` }, beginAtZero:true },
      }
    }
  });

  // Same log-scale and indexed-growth pattern as the by-word charts above,
  // applied per source instead of per word.
  const textsBySource = s => DATA.all_years.map(y => (DATA.source_year_counts[s]||{})[y] || 0);

  new Chart(document.getElementById('chart-source-log'), {
    type: 'line',
    data: {
      labels: DATA.all_years,
      datasets: [
        {
          label: 'All articles',
          data: DATA.all_years.map(y => DATA.survey_by_year[y] || null),
          borderColor: '#c8c8c8', backgroundColor: '#c8c8c8',
          spanGaps: false, tension: 0.25, pointRadius: 1.5, borderWidth: 2,
        },
        ...sources.map((s, i) => ({
          label: s,
          data: textsBySource(s).map(v => v || null),
          borderColor: color(i), backgroundColor: color(i),
          spanGaps: false, tension: 0.25, pointRadius: 1.5, borderWidth: 1.5,
        })),
      ]
    },
    options: {
      plugins: { legend: { position:'right', labels:{boxWidth:10,padding:8,font:{size:10}} } },
      scales: {
        x: { grid:GRID, ticks:TICKS },
        y: { type:'logarithmic', grid:GRID, ticks:TICKS },
      }
    }
  });

  (function(){
    const indexFrom = series => {
      const baseIdx = series.findIndex(v => v > 0);
      if (baseIdx === -1) return null;
      const base = series[baseIdx];
      return series.map((v, i) => i < baseIdx ? null : Math.round(v / base * 1000) / 10);
    };

    const allSeries  = DATA.all_years.map(y => DATA.survey_by_year[y] || 0);
    const datasets   = [];
    const allIndexed = indexFrom(allSeries);
    if (allIndexed) {
      datasets.push({
        label: 'All articles', data: allIndexed,
        borderColor: '#c8c8c8', backgroundColor: '#c8c8c8',
        tension: 0.25, pointRadius: 1.5, borderWidth: 2,
      });
    }
    sources.forEach((s, i) => {
      const sourceIndexed = indexFrom(textsBySource(s));
      if (!sourceIndexed) return;
      datasets.push({
        label: s, data: sourceIndexed,
        borderColor: color(i), backgroundColor: color(i),
        tension: 0.25, pointRadius: 1.5, borderWidth: 1.5,
      });
    });

    new Chart(document.getElementById('chart-source-indexed'), {
      type: 'line',
      data: { labels: DATA.all_years, datasets },
      options: {
        plugins: { legend: { position:'right', labels:{boxWidth:10,padding:8,font:{size:10}} } },
        scales: {
          x: { grid:GRID, ticks:TICKS },
          y: { grid:GRID, ticks:TICKS, beginAtZero:true },
        }
      }
    });
  })();
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
    labels: DATA.full_years,
    datasets: [{ data: DATA.full_years.map(y=>DATA.survey_by_year_full[y]||0), backgroundColor:'#c8c8c8', borderWidth:0 }]
  },
  options: { plugins:{legend:{display:false}}, scales:{ x:{grid:GRID,ticks:TICKS}, y:{grid:GRID,ticks:TICKS,beginAtZero:true} } }
});

// Textile Metaphor texts per year as a percentage of that year's total
// articles — a line, not bars, since it's a rate rather than a count, and
// null (not 0) for years with no survey denominator so the line breaks
// instead of implying a real zero.
new Chart(document.getElementById('chart-year-rate'), {
  type: 'line',
  data: {
    labels: DATA.all_years,
    datasets: [{
      data: DATA.all_years.map(y => {
        const total = DATA.survey_by_year[y] || 0;
        return total > 0 ? (DATA.year_counts[y]||0) / total * 100 : null;
      }),
      borderColor: '#2563a8',
      backgroundColor: 'rgba(37,99,168,0.08)',
      fill: true,
      spanGaps: false,
      tension: 0.25,
      pointRadius: 3,
      pointBackgroundColor: '#2563a8',
    }]
  },
  options: {
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: ctx => `${String(Math.round(ctx.parsed.y*10)/10)}%` } },
    },
    scales: {
      x: { grid:GRID, ticks:TICKS },
      y: { grid:GRID, ticks:{ ...TICKS, callback: v => `${v}%` }, beginAtZero:true },
    }
  }
});

// Both series on one logarithmic axis: a linear axis would flatten the
// smaller Textile Metaphor series to invisibility next to total article
// counts, but a shared log axis makes their growth curves comparable. A
// value of 0 has no logarithm, so zero-count years become null (a gap).
new Chart(document.getElementById('chart-year-log'), {
  type: 'line',
  data: {
    labels: DATA.all_years,
    datasets: [
      {
        label: 'All articles',
        data: DATA.all_years.map(y => DATA.survey_by_year[y] || null),
        borderColor: '#c8c8c8',
        backgroundColor: '#c8c8c8',
        spanGaps: false, tension: 0.25, pointRadius: 2,
      },
      {
        label: 'Textile Metaphor texts',
        data: DATA.all_years.map(y => DATA.year_counts[y] || null),
        borderColor: '#2563a8',
        backgroundColor: '#2563a8',
        spanGaps: false, tension: 0.25, pointRadius: 2,
      },
    ]
  },
  options: {
    plugins: { legend: { position:'top', align:'end', labels:{boxWidth:10,padding:8,font:{size:10}} } },
    scales: {
      x: { grid:GRID, ticks:TICKS },
      y: { type:'logarithmic', grid:GRID, ticks:TICKS },
    }
  }
});

// Both series rebased to 100 at their first year with data for both, so
// relative growth is readable on a plain linear axis (an alternative to the
// log chart above that doesn't require reading a log scale).
(function(){
  const years   = DATA.all_years;
  const textile = years.map(y => DATA.year_counts[y] || 0);
  const all     = years.map(y => DATA.survey_by_year[y] || 0);
  const baseIdx = years.findIndex((_, i) => textile[i] > 0 && all[i] > 0);
  if (baseIdx === -1) return;

  document.getElementById('chart-note-baseline-year').textContent = years[baseIdx];

  const index = (series, base) => series.map(v => Math.round(v / base * 1000) / 10);

  new Chart(document.getElementById('chart-year-indexed'), {
    type: 'line',
    data: {
      labels: years,
      datasets: [
        {
          label: 'All articles',
          data: index(all, all[baseIdx]),
          borderColor: '#c8c8c8', backgroundColor: '#c8c8c8',
          tension: 0.25, pointRadius: 2,
        },
        {
          label: 'Textile Metaphor texts',
          data: index(textile, textile[baseIdx]),
          borderColor: '#2563a8', backgroundColor: '#2563a8',
          tension: 0.25, pointRadius: 2,
        },
      ]
    },
    options: {
      plugins: { legend: { position:'top', align:'end', labels:{boxWidth:10,padding:8,font:{size:10}} } },
      scales: {
        x: { grid:GRID, ticks:TICKS },
        y: { grid:GRID, ticks:TICKS, beginAtZero:true },
      }
    }
  });
})();

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

// Per-word (not aggregated) versions of the rate/log/indexed pattern used
// above for texts-per-year: one line per textile word, plus "All articles"
// as a grey context line where the chart type allows a shared basis.
const hitsByWord = w => DATA.all_years.map(y => (DATA.year_textile[y] || {})[w] || 0);

new Chart(document.getElementById('chart-year-hits-rate'), {
  type: 'line',
  data: {
    labels: DATA.all_years,
    datasets: DATA.textile_words.map((w, i) => {
      const hits = hitsByWord(w);
      return {
        label: w,
        data: DATA.all_years.map((y, idx) => {
          const total = DATA.survey_by_year[y] || 0;
          return total > 0 ? hits[idx] / total * 100 : null;
        }),
        borderColor: color(i), backgroundColor: color(i),
        spanGaps: false, tension: 0.25, pointRadius: 1.5, borderWidth: 1.5,
      };
    })
  },
  options: {
    plugins: {
      legend: { position:'right', labels:{boxWidth:10,padding:8,font:{size:10}} },
      tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${String(Math.round(ctx.parsed.y*10)/10)}%` } },
    },
    scales: {
      x: { grid:GRID, ticks:TICKS },
      y: { grid:GRID, ticks:{ ...TICKS, callback: v => `${v}%` }, beginAtZero:true },
    }
  }
});

new Chart(document.getElementById('chart-year-hits-log'), {
  type: 'line',
  data: {
    labels: DATA.all_years,
    datasets: [
      {
        label: 'All articles',
        data: DATA.all_years.map(y => DATA.survey_by_year[y] || null),
        borderColor: '#c8c8c8', backgroundColor: '#c8c8c8',
        spanGaps: false, tension: 0.25, pointRadius: 1.5, borderWidth: 2,
      },
      ...DATA.textile_words.map((w, i) => ({
        label: w,
        data: hitsByWord(w).map(v => v || null),
        borderColor: color(i), backgroundColor: color(i),
        spanGaps: false, tension: 0.25, pointRadius: 1.5, borderWidth: 1.5,
      })),
    ]
  },
  options: {
    plugins: { legend: { position:'right', labels:{boxWidth:10,padding:8,font:{size:10}} } },
    scales: {
      x: { grid:GRID, ticks:TICKS },
      y: { type:'logarithmic', grid:GRID, ticks:TICKS },
    }
  }
});

// Each series rebased to 100 in its own first year with a nonzero value —
// not one shared baseline year — since different words enter the corpus in
// different years. Years before a series' own baseline are left null.
(function(){
  const indexFrom = series => {
    const baseIdx = series.findIndex(v => v > 0);
    if (baseIdx === -1) return null;
    const base = series[baseIdx];
    return series.map((v, i) => i < baseIdx ? null : Math.round(v / base * 1000) / 10);
  };

  const allSeries = DATA.all_years.map(y => DATA.survey_by_year[y] || 0);
  const datasets  = [];
  const allIndexed = indexFrom(allSeries);
  if (allIndexed) {
    datasets.push({
      label: 'All articles', data: allIndexed,
      borderColor: '#c8c8c8', backgroundColor: '#c8c8c8',
      tension: 0.25, pointRadius: 1.5, borderWidth: 2,
    });
  }
  DATA.textile_words.forEach((w, i) => {
    const wordIndexed = indexFrom(hitsByWord(w));
    if (!wordIndexed) return;
    datasets.push({
      label: w, data: wordIndexed,
      borderColor: color(i), backgroundColor: color(i),
      tension: 0.25, pointRadius: 1.5, borderWidth: 1.5,
    });
  });

  new Chart(document.getElementById('chart-year-hits-indexed'), {
    type: 'line',
    data: { labels: DATA.all_years, datasets },
    options: {
      plugins: { legend: { position:'right', labels:{boxWidth:10,padding:8,font:{size:10}} } },
      scales: {
        x: { grid:GRID, ticks:TICKS },
        y: { grid:GRID, ticks:TICKS, beginAtZero:true },
      }
    }
  });
})();

// ── Frequency (horizontal bar) ────────────────────────────────────
function makeFreq(canvasId, freqData, { percent = false } = {}) {
  new Chart(document.getElementById(canvasId), {
    type: 'bar',
    data: {
      labels: freqData.map(([w])=>w),
      datasets: [{ data: freqData.map(([,n])=>n), backgroundColor: freqData.map((_,i)=>color(i)), borderWidth:0 }]
    },
    options: {
      indexAxis:'y',
      plugins:{
        legend:{display:false},
        tooltip: percent ? { callbacks:{ label: ctx => `${String(ctx.parsed.x)}%` } } : {},
      },
      scales:{
        x:{ grid:GRID, ticks: percent ? { ...TICKS, callback: v => `${v}%` } : TICKS, beginAtZero:true },
        y:{ grid:{display:false}, ticks:{...TICKS,font:{family:"'JetBrains Mono',monospace",size:11}} }
      }
    }
  });
}
makeFreq('chart-textile-freq', DATA.textile_freq);

// Each word's hits as a share of Textile Metaphor texts in the corpus —
// the relative counterpart to the absolute frequency chart above.
makeFreq('chart-textile-freq-rel', DATA.textile_freq.map(([w, n]) => [
  w, DATA.textile_metaphor_texts > 0 ? Math.round(n / DATA.textile_metaphor_texts * 1000) / 10 : 0,
]), { percent: true });

// ── Per-word temporal with tabs ────────────────────────────────────
function makeWordTime(canvasId, tabGroupId, yearData, words, { rate = false } = {}) {
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
        label: word,
        data: DATA.all_years.map(y => {
          const raw = (yearData[y]||{})[word] || 0;
          if (!rate) return raw;
          const total = DATA.survey_by_year[y] || 0;
          return total > 0 ? Math.round(raw / total * 1000) / 10 : null;
        }),
        backgroundColor:'#0a0a0a', borderWidth:0,
      }]},
      options:{
        plugins:{
          legend:{display:false},
          tooltip: rate ? { callbacks:{ label: ctx => `${String(ctx.parsed.y)}%` } } : {},
        },
        scales:{
          x:{grid:GRID,ticks:TICKS},
          y:{grid:GRID, ticks: rate ? { ...TICKS, callback: v => `${v}%` } : TICKS, beginAtZero:true}
        }
      }
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
makeWordTime('chart-textile-word-time-rel','tabs-textile-rel', DATA.year_textile, DATA.textile_words, { rate: true });

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

// ── PNG download for every chart ────────────────────────────────
// One button per chart-wrap's canvas. Reads the canvas at click time (not
// at setup time), so it always exports whatever is currently drawn —
// including charts whose canvas is redrawn on tab clicks (makeWordTime,
// makeTrendChart), which share one canvas across several rendered charts.
// Filenames are {section}_{chart}[_{active tab}...] — the same scheme
// export_chart_images() in dashboard.py uses for the images/ folder, so a
// browser download and its images/ counterpart always share a name.
function slugify(text) {
  return (text || '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
}
document.querySelectorAll('.chart-wrap').forEach(wrap => {
  if (wrap.closest('#cooccurrence')) return; // no PNG export for Co-occurrence & Collocation charts
  const canvas = wrap.querySelector('canvas');
  if (!canvas) return;
  const btn = document.createElement('a');
  btn.className   = 'chart-download';
  btn.href        = 'javascript:;';
  btn.textContent = 'Download PNG';
  btn.addEventListener('click', () => {
    const section     = wrap.closest('section');
    const sectionSlug = slugify(section?.querySelector('.section-title')?.textContent);
    const chartSlug   = slugify(wrap.querySelector('.chart-title')?.textContent);
    const tabSlugs    = [...wrap.querySelectorAll('.tab-group')]
      .map(g => slugify(g.querySelector('.tab-btn.active')?.textContent))
      .filter(Boolean);
    const filename = [sectionSlug, chartSlug, ...tabSlugs].filter(Boolean).join('_') || 'chart';
    const a = document.createElement('a');
    a.href     = canvas.toDataURL('image/png');
    a.download = `${filename}.png`;
    a.click();
  });
  wrap.appendChild(btn);
});
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# 7.  Chart image export
# ---------------------------------------------------------------------------
#
# Renders the just-written dashboard HTML in a headless browser and exports
# every chart as a PNG into IMAGES_DIR, including every tab combination of
# "select word"/"select source" charts (each is redrawn and captured in
# turn). Filenames match the in-page "Download PNG" button exactly:
# {section}_{chart}[_{active tab}...].png — see the JS slugify()/click
# handler in HTML_TEMPLATE. Requires Playwright with a Chromium build
# (`pip install playwright && playwright install chromium`); skipped with a
# warning, not a hard failure, if that isn't available, since the HTML
# dashboard and statistics.csv are the more essential outputs.

IMAGES_DIR = Path("../images")
CHART_ANIMATION_WAIT_MS = 1200  # >= Chart.js's default 1000ms animation duration

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")

def _save_canvas_png(page, canvas_selector: str, path: Path) -> None:
    data_url = page.eval_on_selector(canvas_selector, "el => el.toDataURL('image/png')")
    _, encoded = data_url.split(",", 1)
    path.write_bytes(base64.b64decode(encoded))

def _export_tab_combinations(page, wrap_selector: str, group_index: int,
                              tab_group_ids: list[str], name_parts: list[str],
                              images_dir: Path, written: list[int]) -> None:
    """
    Recurse over every tab group in a chart-wrap (in DOM order), clicking
    each button and exporting once all groups are exhausted. Later tab
    groups (e.g. the lemma tabs in a word→lemma co-occurrence chart) are
    only queried after their preceding group's click has regenerated them.
    """
    if group_index >= len(tab_group_ids):
        filename = "_".join(name_parts) or "chart"
        _save_canvas_png(page, f"{wrap_selector} canvas", images_dir / f"{filename}.png")
        written[0] += 1
        return

    group_selector = f'#{tab_group_ids[group_index]}'
    btn_count = page.eval_on_selector_all(f"{group_selector} .tab-btn", "els => els.length")
    for i in range(btn_count):
        btn_selector = f"{group_selector} .tab-btn:nth-of-type({i + 1})"
        btn_text = page.eval_on_selector(btn_selector, "el => el.textContent")
        page.click(btn_selector)
        page.wait_for_timeout(CHART_ANIMATION_WAIT_MS)
        _export_tab_combinations(page, wrap_selector, group_index + 1, tab_group_ids,
                                  name_parts + [_slug(btn_text)], images_dir, written)

def export_chart_images(html_path: Path, images_dir: Path) -> int:
    """Export every chart in html_path to images_dir as PNGs. Returns the
    number of images written, or 0 if Playwright/Chromium isn't available."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  NOTE: Playwright not installed — skipping images/ export. "
              "Run: pip install playwright && playwright install chromium")
        return 0

    images_dir.mkdir(exist_ok=True)
    for old_png in images_dir.glob("*.png"):
        old_png.unlink()

    written = [0]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.goto(html_path.resolve().as_uri())
            page.wait_for_selector(".chart-wrap canvas")
            page.wait_for_timeout(CHART_ANIMATION_WAIT_MS)

            # Co-occurrence & Collocation charts are excluded from image export
            # (same as their in-page "Download PNG" buttons) — skipped here,
            # with the remaining chart-wraps re-indexed contiguously from 0.
            page.evaluate("""
                (() => {
                    let i = 0;
                    document.querySelectorAll('.chart-wrap').forEach(el => {
                        if (!el.closest('#cooccurrence')) el.setAttribute('data-chart-index', i++);
                    });
                })();
            """)
            wrap_count = page.eval_on_selector_all(".chart-wrap[data-chart-index]", "els => els.length")

            for i in range(wrap_count):
                wrap_selector = f'.chart-wrap[data-chart-index="{i}"]'
                info = page.eval_on_selector(wrap_selector, """el => ({
                    section: el.closest('section')?.querySelector('.section-title')?.textContent || '',
                    chart: el.querySelector('.chart-title')?.textContent || '',
                })""")
                base_parts = [p for p in (_slug(info["section"]), _slug(info["chart"])) if p]
                tab_group_ids = page.eval_on_selector_all(
                    f"{wrap_selector} .tab-group", "els => els.map(e => e.id).filter(Boolean)"
                )
                _export_tab_combinations(page, wrap_selector, 0, tab_group_ids,
                                          base_parts, images_dir, written)
                print(f"  ...{written[0]} images written so far", end="\r")

            browser.close()
    except Exception as exc:
        print(f"\n  NOTE: chart image export failed ({exc}); "
              f"{written[0]} image(s) written before the error.")

    print()
    return written[0]

# ---------------------------------------------------------------------------
# 8.  Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CLEAN_CSV_PATH.exists():
        sys.exit(f"ERROR: Clean CSV not found: {CLEAN_CSV_PATH}")

    print(f"Loading {CLEAN_CSV_PATH} …")
    clean_rows = load_csv(CLEAN_CSV_PATH)
    print(f"  {len(clean_rows)} Textile Metaphor occurrence rows.")

    print("Computing statistics …")
    stats = build_stats(clean_rows)
    print(f"  Done. {stats['textile_metaphor_texts']} texts, "
          f"{len(stats['textile_words'])} textile words.")

    if JOURNAL_COUNTS_PATH.exists():
        print(f"Loading {JOURNAL_COUNTS_PATH} …")
        journal_counts = load_journal_counts(JOURNAL_COUNTS_PATH)
        stats.update(compute_survey_coverage(clean_rows, journal_counts, stats["all_years"]))
        print(f"  Articles surveyed: {stats['articles_surveyed_total']}, "
              f"Textile Metaphor rate: {stats['textile_metaphor_rate_pct']}%.")
    else:
        print(f"  NOTE: {JOURNAL_COUNTS_PATH} not found — skipping survey-coverage statistics.")
        stats["articles_surveyed_total"]   = None
        stats["textile_metaphor_rate_pct"] = None
        stats["survey_by_year"]            = {}
        stats["survey_by_source"]          = {}
        stats["survey_by_source_year"]     = {}
        stats["full_years"]                = []
        stats["survey_by_year_full"]       = {}

    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json.dumps(stats, ensure_ascii=False))
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"\nDashboard written to: {OUTPUT_PATH}")

    write_statistics_csv(stats, STATISTICS_CSV_PATH)
    print(f"Statistics written to: {STATISTICS_CSV_PATH}")

    print(f"\nExporting chart images to {IMAGES_DIR}/ (every chart, every tab) …")
    image_count = export_chart_images(OUTPUT_PATH, IMAGES_DIR)
    if image_count:
        print(f"{image_count} images written to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()