# MGPHot Browser (webapp)

A minimal FastAPI web application to browse the MGPHot dataset, visualize gene values, and traverse to similar items by gene profile. Where available, the app also surfaces context pulled from Wikidata based on pre-aligned MusicBrainz identifiers.

## Features

- Tabular gene view per item with categories and descriptions from `mgphot_genes.tsv`.
- Similarity traversal using k-NN over numeric gene vectors.
- Wikidata context (artist/work) using QIDs from precomputed `aligned.tsv` (if present). No string searches are performed.
- Simple, clean UI with keyboard-like prev/next navigation.

## Run locally

1. Create a virtualenv and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r webapp/requirements.txt
```

2. Start the server from the `webapp` folder:

```bash
uvicorn app:app --reload --port 8000
```

3. Open http://127.0.0.1:8000

## Data expectations

- `mgphot_genes.tsv` and `mgphot_gene_values.tsv` should be at the repo root (one level up from `webapp/`).
- Optionally, `aligned.tsv` at the repo root can provide MusicBrainz/Wikidata IDs to enrich item pages.

## Notes

- This is WIP and not guaranteed to be correct. Validate before relying on results.
- For production use, add caching for Wikidata entity fetches and consider persisting a joined dataset for speed.
