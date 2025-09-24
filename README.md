# MGPHot Enrichment (MusicBrainz → Wikidata bridge)

This project is a first step toward enriching the MGPHot dataset presented by Sergio Oramas and colleagues at ISMIR 2025 with contextual metadata, using MusicBrainz as a bridge to Wikidata.

- MGPHot dataset: https://doi.org/10.5281/zenodo.15372063
- Goal: For each chart row (artist, title), align to a MusicBrainz Recording MBID, derive the Artist MBID and Work MBID (when available), and resolve corresponding Wikidata QIDs via exact property links (no name/title string search).

## What it does

- Searches MusicBrainz for recordings using artist + title, with normalization and fallbacks.
- Uses fuzzy title distance scoring to select the best candidate; if multiple plausible matches exist, it returns the most promising candidate and flags the row as "ambiguous".
- When a recording MBID is found, looks up:
  - Artist MBID(s) from the recording's artist-credit.
  - Work MBID(s) from work relationships (if present).
- Queries Wikidata strictly by MusicBrainz identifiers (no string lookups):
  - Artist QID via property P434 (MusicBrainz artist ID).
  - Work QID via property P435 (MusicBrainz work ID).

## Status and disclaimer

This is work in progress, vibecoded using Copilot / GPT-5. No claims are made for accuracy or correctness. Expect rough edges, incomplete coverage, and misalignments. Please validate and do not rely on results.

## Quick start

- Requirements: Python 3.9+ and internet access.
- Main script: `align_mgphot_to_mb.py`

Examples:

```bash
# Align the full input (tab-separated), write to stdout
python3 align_mgphot_to_mb.py --input hot100_charts.tsv --output -

# Sample N random rows with a fixed seed for reproducibility
python3 align_mgphot_to_mb.py --sample 20 --seed 1234 --input hot100_charts.tsv --output sample-20.1234.tsv

# Add candidates JSON column for ambiguous/not_found cases
python3 align_mgphot_to_mb.py --emit-candidates --input hot100_charts.tsv --output aligned.tsv
```

Output columns:

- chart_week
- artist
- title
- mgphot_track_id
- mb_recording_mbid
- mb_artist_mbid
- wd_artist_qid
- mb_work_mbid
- wd_work_qid
- status (matched | ambiguous | not_found | error)
- candidates_json (optional; only for ambiguous/not_found)

Notes:
- The script rate-limits requests to MusicBrainz and Wikidata and caches responses in a simple local JSON cache.
- Please respect MusicBrainz and Wikidata usage guidelines.

## Data sources and identifiers

- MusicBrainz Web Service 2: https://musicbrainz.org/doc/MusicBrainz_API
- Wikidata Query Service (SPARQL): https://query.wikidata.org/
- Wikidata properties used:
  - P434 (MusicBrainz artist ID)
  - P435 (MusicBrainz work ID)

## License

MIT License

Copyright (c) 2025 David M. Weigl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
