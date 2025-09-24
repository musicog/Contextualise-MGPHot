#!/usr/bin/env python3
"""
Align MGPHot track identifiers to MusicBrainz Recording MBIDs.

Reads an input TSV (default: hot100_charts.tsv) with at least the columns:
  - chart_week
  - artist
  - title
  - mgphot_track_id

For each row, performs a MusicBrainz recording search using artist + title.
Attempts to find a unique recording MBID. If multiple plausible candidates or none
are found, emits a warning to stderr.

Output (TSV to stdout by default):
chart_week	artist	title	mgphot_track_id	mb_recording_mbid	status

status values:
  matched          - unique high-confidence alignment (exact or clearly best)
  ambiguous        - multiple plausible or fuzzy near match; best candidate MBID still reported
  not_found        - no plausible match
  error            - request or parsing error

Usage:
  python align_mgphot_to_mb.py --input hot100_charts.tsv > aligned.tsv 2>warnings.log

Rate limiting: Respects MusicBrainz guidelines (~1 request/sec). Implements simple in-memory cache.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
import json
import re
import hashlib
import difflib
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import quote_plus
import urllib.request
import urllib.error
import atexit
import threading
import asyncio
import random

MB_BASE = "https://musicbrainz.org/ws/2"
USER_AGENT = "MGPHotAlignment/0.2 ( research use )"
REQUEST_INTERVAL = 1.1  # seconds between uncached requests
WD_REQUEST_INTERVAL = 0.8  # seconds between uncached Wikidata requests

TITLE_CLEAN_RE = re.compile(r"\s+|[‘’]|")
PARENS_RE = re.compile(r"\s*\([^)]*\)")
FEAT_RE = re.compile(r"\s+feat\.?\s+.*$", re.IGNORECASE)
NONWORD_RE = re.compile(r"[^A-Za-z0-9]+")

@dataclass
class MBRecordingCandidate:
    id: str
    title: str
    score: int
    artist: str
    length: Optional[int]
    disambiguation: Optional[str]
    first_release_year: Optional[int] = None


def normalize_text(s: str) -> str:
    s0 = s.strip()
    s1 = FEAT_RE.sub("", s0)
    s2 = PARENS_RE.sub("", s1)
    s3 = s2.lower()
    s4 = NONWORD_RE.sub(" ", s3)
    return " ".join(s4.split())


def build_query(artist: str, title: str) -> str:
    """Build a strict Lucene query for recording search."""
    a_norm = artist.strip()
    t_norm = title.strip()
    return f"recording:(\"{t_norm}\") AND artist:(\"{a_norm}\")"

def build_loose_query(artist: str, title: str) -> str:
    """Softer query: just terms without explicit field coupling for fallback."""
    a = normalize_text(artist).replace(" ", " AND ")
    t = normalize_text(title).replace(" ", " AND ")
    return f"{t} AND {a}"

_cache: Dict[str, Any] = {}
_last_request_time = 0.0
_last_wd_request_time = 0.0
_cache_lock = threading.Lock()


def mb_get(path: str, params: Dict[str, str]) -> Any:
    global _last_request_time
    qp = "&".join(f"{k}={quote_plus(v)}" for k, v in params.items())
    url = f"{MB_BASE}/{path}?{qp}"
    with _cache_lock:
        if url in _cache:
            return _cache[url]
    # rate limit
    now = time.time()
    dt = now - _last_request_time
    if dt < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - dt)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
            with _cache_lock:
                _cache[url] = data
            _last_request_time = time.time()
            return data
    except urllib.error.HTTPError as e:
        print(f"WARNING\tHTTP {e.code}\t{url}", file=sys.stderr)
    except Exception as e:  # noqa
        print(f"WARNING\tRequestError\t{e}\t{url}", file=sys.stderr)
    return None

def mb_lookup_recording(recording_mbid: str, inc: str = "artist-credits+work-rels") -> Any:
    """Lookup a recording by MBID including specified inc expansions."""
    params = {"fmt": "json", "inc": inc}
    return mb_get(f"recording/{recording_mbid}", params)

def extract_artist_mbids_from_recording(rec_json: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for ac in rec_json.get("artist-credit", []) or []:
        art = ac.get("artist") or {}
        aid = art.get("id")
        if aid:
            ids.append(aid)
    return ids

def extract_work_mbids_from_recording(rec_json: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for rel in rec_json.get("relations", []) or []:
        # Work relationships typically have a 'work' key when inc=work-rels
        work = rel.get("work")
        if work and isinstance(work, dict):
            wid = work.get("id")
            if wid:
                ids.append(wid)
    return ids

def wd_sparql(query: str) -> Any:
    """Execute a SPARQL GET against Wikidata with simple rate limiting and caching."""
    global _last_wd_request_time
    base = "https://query.wikidata.org/sparql"
    qp = f"query={quote_plus(query)}&format=json"
    url = f"{base}?{qp}"
    with _cache_lock:
        if url in _cache:
            return _cache[url]
    # rate limit
    now = time.time()
    dt = now - _last_wd_request_time
    if dt < WD_REQUEST_INTERVAL:
        time.sleep(WD_REQUEST_INTERVAL - dt)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
            with _cache_lock:
                _cache[url] = data
            _last_wd_request_time = time.time()
            return data
    except urllib.error.HTTPError as e:
        print(f"WARNING\tWD_HTTP {e.code}\t{url}", file=sys.stderr)
    except Exception as e:  # noqa
        print(f"WARNING\tWD_RequestError\t{e}\t{url}", file=sys.stderr)
    return None

def wd_find_item_by_property(prop: str, mbid: str) -> Optional[str]:
    """Return QID for item having wdt:prop == mbid, if any."""
    # Escape quotes inside mbid if any
    mbid_lit = mbid.replace('"', '\\"')
    sparql = f"SELECT ?item WHERE {{ ?item wdt:{prop} \"{mbid_lit}\" . }} LIMIT 1"
    data = wd_sparql(sparql)
    try:
        for b in data.get("results", {}).get("bindings", []):
            uri = b.get("item", {}).get("value")
            if uri and uri.startswith("http://www.wikidata.org/entity/"):
                return uri.rsplit("/", 1)[-1]
    except Exception:
        pass
    return None

def get_enrichment_for_recording(recording_mbid: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Given a recording MBID, return (artist_mbid, artist_qid, work_mbid, work_qid). Picks first artist/work if multiple."""
    rec = mb_lookup_recording(recording_mbid)
    if not rec:
        return (None, None, None, None)
    artist_mbids = extract_artist_mbids_from_recording(rec)
    work_mbids = extract_work_mbids_from_recording(rec)
    artist_mbid = artist_mbids[0] if artist_mbids else None
    work_mbid = work_mbids[0] if work_mbids else None
    artist_qid = wd_find_item_by_property("P434", artist_mbid) if artist_mbid else None
    work_qid = wd_find_item_by_property("P435", work_mbid) if work_mbid else None
    return (artist_mbid, artist_qid, work_mbid, work_qid)


def parse_recordings(data: Any) -> List[MBRecordingCandidate]:
    out: List[MBRecordingCandidate] = []
    if not data or "recordings" not in data:
        return out
    for rec in data.get("recordings", []):
        rid = rec.get("id")
        rtitle = rec.get("title", "")
        score = int(rec.get("score", 0))
        length = rec.get("length")
        disamb = rec.get("disambiguation")
        artist_credit = " ".join(ac.get("name", "") for ac in rec.get("artist-credit", []))
        # Attempt earliest release year if releases included
        year: Optional[int] = None
        for rel in rec.get("releases", []) or []:
            date = rel.get("date")
            if date and len(date) >= 4 and date[:4].isdigit():
                y = int(date[:4])
                if year is None or y < year:
                    year = y
        out.append(MBRecordingCandidate(rid, rtitle, score, artist_credit, length, disamb, year))
    return out


def search_recordings(artist: str, title: str, limit: int = 10) -> List[MBRecordingCandidate]:
    # First attempt strict query
    params = {"query": build_query(artist, title), "fmt": "json", "limit": str(limit), "inc": "releases"}
    data = mb_get("recording", params)
    cands = parse_recordings(data)
    if cands:
        return cands
    # Fallback loose query
    params_loose = {"query": build_loose_query(artist, title), "fmt": "json", "limit": str(limit), "inc": "releases"}
    data2 = mb_get("recording", params_loose)
    return parse_recordings(data2)


def pick_unique(cands: List[MBRecordingCandidate], artist: str, title: str, target_year: Optional[int]) -> Tuple[str, str]:
    """Select best candidate with fuzzy fallback.

    Returns (mbid, status) where status is one of:
      matched   - high confidence unique or clearly superior candidate
      ambiguous - multiple plausible (close MB scores or fuzzy near matches); best MBID still reported
      not_found - nothing acceptable
    """
    if not cands:
        return ("", "not_found")

    n_title = normalize_text(title)
    artist_tokens = set(normalize_text(artist).split())

    def artist_overlap(c: MBRecordingCandidate) -> bool:
        cand_tokens = set(normalize_text(c.artist).split())
        return bool(artist_tokens & cand_tokens)

    # 1. Exact normalized title matches (artist overlap filtered)
    exact = [c for c in cands if normalize_text(c.title) == n_title]
    exact = [c for c in exact if artist_overlap(c)] or exact
    if exact:
        if target_year is not None:
            exact.sort(key=lambda c: (abs((c.first_release_year or target_year) - target_year), -c.score, c.id))
        else:
            exact.sort(key=lambda c: (-c.score, c.id))
        top = exact[0]
        ambiguous_exact = False
        if len(exact) > 1:
            second = exact[1]
            if second.score == top.score:
                ambiguous_exact = True
            else:
                close = [c for c in exact if c.score >= top.score - 5]
                if len(close) > 1 and close[0].id != top.id:
                    ambiguous_exact = True
        if ambiguous_exact:
            return (top.id, "ambiguous")
        return (top.id, "matched")

    # 2. High MB score fallback (>=95)
    high = [c for c in cands if c.score >= 95 and artist_overlap(c)]
    if target_year is not None and len(high) > 1:
        high.sort(key=lambda c: (abs((c.first_release_year or target_year) - target_year), -c.score, c.id))
        if len(high) >= 2:
            d0 = abs((high[0].first_release_year or target_year) - target_year)
            d1 = abs((high[1].first_release_year or target_year) - target_year)
            if d0 < d1:
                return (high[0].id, "matched")
    if len(high) == 1:
        return (high[0].id, "matched")
    if len(high) > 1:
        high.sort(key=lambda c: (-c.score, c.id))
        return (high[0].id, "ambiguous")

    # 3. Fuzzy similarity (SequenceMatcher ratio) with artist overlap
    fuzzy_scores: List[Tuple[float, MBRecordingCandidate]] = []
    for c in cands:
        if not artist_overlap(c):
            continue
        cand_norm = normalize_text(c.title)
        if not cand_norm:
            continue
        ratio = difflib.SequenceMatcher(None, n_title, cand_norm).ratio() * 100.0
        fuzzy_scores.append((ratio, c))
    if fuzzy_scores:
        fuzzy_scores.sort(key=lambda rc: (-rc[0], -rc[1].score, rc[1].id))
        best_ratio, best_cand = fuzzy_scores[0]
        FUZZY_MIN = 82.0
        if best_ratio >= FUZZY_MIN:
            if len(fuzzy_scores) > 1:
                second_ratio = fuzzy_scores[1][0]
                gap = best_ratio - second_ratio
            else:
                second_ratio = 0.0
                gap = best_ratio
            if best_ratio >= 97.0 and gap >= 5.0:
                return (best_cand.id, "matched")
            return (best_cand.id, "ambiguous")

    return ("", "not_found")


def extract_year(chart_week: str) -> Optional[int]:
    if chart_week and len(chart_week) >= 4 and chart_week[:4].isdigit():
        try:
            y = int(chart_week[:4])
            if 1900 <= y <= 2100:
                return y
        except ValueError:
            return None
    return None


def process_rows(reader: csv.DictReader, out_writer: csv.writer, limit: Optional[int] = None, emit_candidates: bool = False):
    count = 0
    header = [
        "chart_week", "artist", "title", "mgphot_track_id",
        "mb_recording_mbid", "mb_artist_mbid", "wd_artist_qid", "mb_work_mbid", "wd_work_qid",
        "status",
    ]
    if emit_candidates:
        header.append("candidates_json")
    out_writer.writerow(header)
    for row in reader:
        chart_week = row.get("chart_week", "")
        artist = row.get("artist", "")
        title = row.get("title", "")
        mgphot_track_id = row.get("mgphot_track_id", "")
        if not artist or not title:
            print(f"WARNING\tMissingFields\t{mgphot_track_id}", file=sys.stderr)
            out_writer.writerow([chart_week, artist, title, mgphot_track_id, "", "", "", "", "", "not_found"])
            continue
        cands = search_recordings(artist, title)
        year = extract_year(chart_week)
        mbid, status = pick_unique(cands, artist, title, year)
        if status != "matched":
            print(f"WARNING\t{status}\t{artist}\t{title}\t{mgphot_track_id}", file=sys.stderr)
        # Enrich via MB/Wikidata when we have a recording MBID
        mb_artist_mbid = wd_artist_qid = mb_work_mbid = wd_work_qid = ""
        if mbid:
            a_mbid, a_qid, w_mbid, w_qid = get_enrichment_for_recording(mbid)
            mb_artist_mbid = a_mbid or ""
            wd_artist_qid = a_qid or ""
            mb_work_mbid = w_mbid or ""
            wd_work_qid = w_qid or ""
        row_out = [chart_week, artist, title, mgphot_track_id, mbid, mb_artist_mbid, wd_artist_qid, mb_work_mbid, wd_work_qid, status]
        if emit_candidates:
            if status in ("ambiguous", "not_found"):
                # Include limited candidate info
                cjson = json.dumps([
                    {
                        "id": c.id,
                        "title": c.title,
                        "score": c.score,
                        "artist": c.artist,
                        "year": c.first_release_year,
                    } for c in cands
                ])
            else:
                cjson = ""
            row_out.append(cjson)
        out_writer.writerow(row_out)
        count += 1
        if limit and count >= limit:
            break


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Align MGPHot chart rows to MusicBrainz recording MBIDs")
    p.add_argument("--input", "-i", default="hot100_charts.tsv", help="Input TSV file path")
    p.add_argument("--output", "-o", default="-", help="Output TSV path or - for stdout")
    p.add_argument("--limit", type=int, default=None, help="Limit number of rows (for testing)")
    p.add_argument("--no-header", action="store_true", help="Input file has no header row")
    p.add_argument("--emit-candidates", action="store_true", help="Add JSON column of candidates for ambiguous/not_found rows")
    p.add_argument("--cache-file", default=".mb_cache.json", help="Path to persistent cache file (JSON)")
    p.add_argument("--async", dest="use_async", action="store_true", help="Experimental: use asyncio implementation")
    p.add_argument("--workers", type=int, default=1, help="Concurrent async tasks (still rate-limited globally)")
    p.add_argument("--sample", type=int, default=None, help="Randomly sample N rows to process (applied before --limit)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    return p.parse_args(argv)


def open_output(path: str):
    if path == "-":
        return sys.stdout
    return open(path, "w", newline="", encoding="utf-8")


def load_cache(path: str):
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            with _cache_lock:
                _cache.update(data)
    except Exception as e:  # noqa
        print(f"WARNING\tCacheLoadFailed\t{e}", file=sys.stderr)


def save_cache(path: str):
    if not path:
        return
    try:
        with _cache_lock:
            data = dict(_cache)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception as e:  # noqa
        print(f"WARNING\tCacheSaveFailed\t{e}", file=sys.stderr)


async def async_process_rows(rows: List[Dict[str, str]], writer: csv.writer, limit: Optional[int], emit_candidates: bool, workers: int):
    # Simple sequential fallback respecting global rate limit; potential future: queue + workers
    # For now just reuse sync logic to avoid duplication.
    class DummyReader(list):
        fieldnames = ["chart_week","artist","title","mgphot_track_id"]
    reader = rows  # we will adapt by manual iteration
    writer.writerow([
        "chart_week", "artist", "title", "mgphot_track_id",
        "mb_recording_mbid", "mb_artist_mbid", "wd_artist_qid", "mb_work_mbid", "wd_work_qid",
        "status",
    ] + (["candidates_json"] if emit_candidates else []))
    count = 0
    for row in rows:
        # Reuse search + pick (which are sync / blocking HTTP); to truly async would need aiohttp
        chart_week = row.get("chart_week", "")
        artist = row.get("artist", "")
        title = row.get("title", "")
        mgphot_track_id = row.get("mgphot_track_id", "")
        if not artist or not title:
            print(f"WARNING\tMissingFields\t{mgphot_track_id}", file=sys.stderr)
            writer.writerow([chart_week, artist, title, mgphot_track_id, "", "", "", "", "", "not_found"] + (["" ] if emit_candidates else []))
            continue
        cands = search_recordings(artist, title)
        year = extract_year(chart_week)
        mbid, status = pick_unique(cands, artist, title, year)
        if status != "matched":
            print(f"WARNING\t{status}\t{artist}\t{title}\t{mgphot_track_id}", file=sys.stderr)
        mb_artist_mbid = wd_artist_qid = mb_work_mbid = wd_work_qid = ""
        if mbid:
            a_mbid, a_qid, w_mbid, w_qid = get_enrichment_for_recording(mbid)
            mb_artist_mbid = a_mbid or ""
            wd_artist_qid = a_qid or ""
            mb_work_mbid = w_mbid or ""
            wd_work_qid = w_qid or ""
        row_out = [chart_week, artist, title, mgphot_track_id, mbid, mb_artist_mbid, wd_artist_qid, mb_work_mbid, wd_work_qid, status]
        if emit_candidates:
            if status in ("ambiguous", "not_found"):
                cjson = json.dumps([{ "id": c.id, "title": c.title, "score": c.score, "artist": c.artist, "year": c.first_release_year } for c in cands])
            else:
                cjson = ""
            row_out.append(cjson)
        writer.writerow(row_out)
        count += 1
        if limit and count >= limit:
            break
        await asyncio.sleep(0)  # allow cooperative switching


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return 2

    # Load cache
    cache_path = args.cache_file
    if cache_path:
        load_cache(cache_path)
        atexit.register(lambda: save_cache(cache_path))

    with open(input_path, "r", encoding="utf-8") as f_in, open_output(args.output) as f_out:
        # Sniff header
        sample_buf = f_in.read(4096)
        f_in.seek(0)
        has_header = not args.no_header
        if has_header:
            reader = csv.DictReader(f_in, delimiter='\t')
        else:
            # Define columns manually
            reader = csv.DictReader(f_in, fieldnames=["chart_week","artist","title","mgphot_track_id"], delimiter='\t')
        writer = csv.writer(f_out, delimiter='\t', lineterminator='\n')

        # Apply optional random sampling before processing
        rows_iterable: List[Dict[str, str]] | Any
        if args.sample is not None:
            rows_all = list(reader)
            k = max(0, min(args.sample, len(rows_all)))
            rng = random.Random(args.seed) if args.seed is not None else random
            rows_iterable = rng.sample(rows_all, k)
        else:
            rows_iterable = reader

        if args.use_async:
            rows_list = list(rows_iterable)
            asyncio.run(async_process_rows(rows_list, writer, args.limit, args.emit_candidates, args.workers))
        else:
            process_rows(rows_iterable, writer, limit=args.limit, emit_candidates=args.emit_candidates)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
