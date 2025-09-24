from __future__ import annotations
import os
import json
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import httpx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_GENES = os.path.join(ROOT_DIR, 'mgphot_genes.tsv')
DATA_GENE_VALUES = os.path.join(ROOT_DIR, 'mgphot_gene_values.tsv')
ALIGNED = os.path.join(ROOT_DIR, 'aligned.tsv')  # optional; may not exist yet
HOT100 = os.path.join(ROOT_DIR, 'hot100_charts.tsv')  # for chart_week display

app = FastAPI(title="MGPHot Browser")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, 'static')), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, 'templates'))

# -- Data loading --
GENE_META: pd.DataFrame
GENE_VALUES: pd.DataFrame
ITEMS: pd.DataFrame

# Neighbors model (by genes)
NN_MODEL: Optional[NearestNeighbors] = None
SCALED: Optional[np.ndarray] = None

async def wd_get_props(artist_qid: Optional[str], work_qid: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=10) as client:
        if artist_qid:
            try:
                r = await client.get(f"https://www.wikidata.org/wiki/Special:EntityData/{artist_qid}.json")
                data = r.json()
                out['artist'] = data
            except Exception:
                out['artist'] = None
        if work_qid:
            try:
                r = await client.get(f"https://www.wikidata.org/wiki/Special:EntityData/{work_qid}.json")
                data = r.json()
                out['work'] = data
            except Exception:
                out['work'] = None
    return out


def load_data():
    global GENE_META, GENE_VALUES, ITEMS, NN_MODEL, SCALED
    # Genes metadata
    GENE_META = pd.read_csv(DATA_GENES, sep='\t')
    # Ensure gene_id is integer and ordered
    if 'gene_id' in GENE_META.columns:
        try:
            GENE_META['gene_id'] = GENE_META['gene_id'].astype(int)
        except Exception:
            pass
    # Per-item gene values
    GENE_VALUES = pd.read_csv(DATA_GENE_VALUES, sep='\t')
    # Normalize mgphot_track_id dtype for safe joins
    if 'mgphot_track_id' in GENE_VALUES.columns:
        GENE_VALUES['mgphot_track_id'] = pd.to_numeric(GENE_VALUES['mgphot_track_id'], errors='coerce').astype('Int64')
    # Parse list column 'gene_values' -> python list[float]
    if 'gene_values' in GENE_VALUES.columns:
        def _parse_vec(v):
            if isinstance(v, list):
                return v
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except Exception:
                    return None
            return None
        GENE_VALUES['gene_vector'] = GENE_VALUES['gene_values'].apply(_parse_vec)
    else:
        # Fallback: attempt to assemble vector from columns named by gene names
        names = list(GENE_META['name']) if 'name' in GENE_META.columns else []
        def _row_to_vec(row):
            return [row.get(n, None) for n in names]
        GENE_VALUES['gene_vector'] = GENE_VALUES.apply(_row_to_vec, axis=1)
    # Optionally bring in a representative chart_week (earliest week) from Hot 100
    if os.path.exists(HOT100):
        try:
            hcols = ['chart_week', 'mgphot_track_id']
            hot = pd.read_csv(HOT100, sep='\t', usecols=hcols)
            # Coerce id to nullable int
            hot['mgphot_track_id'] = pd.to_numeric(hot['mgphot_track_id'], errors='coerce').astype('Int64')
            # Some rows may lack id; drop those for mapping
            hot = hot.dropna(subset=['mgphot_track_id', 'chart_week'])
            # Pick earliest chart week per track id (YYYY-MM-DD sorts lexicographically)
            week_map = (
                hot.groupby('mgphot_track_id', dropna=True)['chart_week']
                .min()
                .reset_index()
            )
            # Left-join onto values
            if 'mgphot_track_id' in GENE_VALUES.columns:
                GENE_VALUES = GENE_VALUES.merge(week_map, on='mgphot_track_id', how='left')
                # Ensure string or empty for template truthiness checks
                GENE_VALUES['chart_week'] = GENE_VALUES['chart_week'].fillna('')
        except Exception:
            # If anything goes wrong, just skip chart week enrichment
            pass

    # Try to merge enrichment if present
    if os.path.exists(ALIGNED):
        aligned = pd.read_csv(ALIGNED, sep='\t')
    else:
        aligned = pd.DataFrame()
    # Keep essential columns and parsed vector
    keep_cols = [c for c in ('mgphot_track_id','artist','title','chart_week','year','gene_vector') if c in GENE_VALUES.columns or c == 'gene_vector']
    ITEMS = GENE_VALUES[keep_cols].copy()
    # Build matrix X from gene_vector lists
    vecs = [v if isinstance(v, list) else [] for v in ITEMS['gene_vector']]
    # Determine expected length from gene meta
    expected_len = int(GENE_META['gene_id'].max() + 1) if 'gene_id' in GENE_META.columns else (len(GENE_META) if len(GENE_META) else (len(vecs[0]) if vecs and vecs[0] else 0))
    def _fix_len(v):
        if not isinstance(v, list):
            return [0.0] * expected_len
        if len(v) < expected_len:
            return v + [0.0] * (expected_len - len(v))
        if len(v) > expected_len:
            return v[:expected_len]
        return v
    vecs = [_fix_len(v) for v in vecs]
    X = np.asarray(vecs, dtype=float) if vecs else np.zeros((0, expected_len), dtype=float)
    if X.shape[0] == 0 or X.shape[1] == 0:
        NN_MODEL = None
        SCALED = None
        return
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    SCALED = Xs
    NN_MODEL = NearestNeighbors(n_neighbors=min(11, len(Xs)), metric='euclidean')
    NN_MODEL.fit(Xs)


@app.on_event("startup")
def _startup():
    load_data()


def get_item(idx: int) -> Dict[str, Any]:
    if idx < 0 or idx >= len(ITEMS):
        raise HTTPException(404, detail="Item index out of range")
    row = ITEMS.iloc[idx].to_dict()
    row['__index__'] = idx
    return row


def neighbors_for(idx: int, k: int = 10) -> List[int]:
    if NN_MODEL is None or SCALED is None:
        return []
    x = SCALED[idx:idx+1, :]
    dists, inds = NN_MODEL.kneighbors(x, n_neighbors=min(k+1, SCALED.shape[0]))
    cand = [int(i) for i in inds[0] if int(i) != idx]
    return cand[:k]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, i: int = 0):
    item = get_item(i)
    # Make a gene table: one row per gene in the order of mgphot_genes.tsv
    gene_rows = []
    # Retrieve vector for this item
    try:
        vec = ITEMS.iloc[item['__index__']].get('gene_vector', None)
    except Exception:
        vec = None
    if not isinstance(vec, list):
        vec = []
    if 'gene_id' in GENE_META.columns:
        meta_iter = GENE_META.sort_values('gene_id')
    else:
        meta_iter = GENE_META
    for _, m in meta_iter.iterrows():
        g_name = m.get('name')
        cat = m.get('category', '')
        desc = m.get('description', '')
        gid = m.get('gene_id') if 'gene_id' in m else None
        try:
            idx_g = int(gid) if pd.notna(gid) else None
        except Exception:
            idx_g = None
        value = None
        if idx_g is not None and idx_g < len(vec):
            value = vec[idx_g]
        gene_rows.append({
            'name': g_name if isinstance(g_name, str) else (str(idx_g) if idx_g is not None else ''),
            'category': cat,
            'description': desc,
            'value': value,
            'gene_id': idx_g
        })
    # Similar items
    neigh_idx = neighbors_for(item['__index__'], k=10)
    neighbors = []
    for j in neigh_idx:
        n = get_item(j)
        try:
            n_vec = ITEMS.iloc[j].get('gene_vector', None)
        except Exception:
            n_vec = None
        n['gene_vector'] = n_vec if isinstance(n_vec, list) else []
        neighbors.append(n)

    # Wikidata context via aligned.tsv if present
    wd_context: Dict[str, Any] = {}
    if os.path.exists(ALIGNED):
        try:
            aligned = pd.read_csv(ALIGNED, sep='\t')
            key_cols = [c for c in ('mgphot_track_id','artist','title','chart_week') if c in aligned.columns and c in ITEMS.columns]
            if key_cols:
                # Best-effort join by mgphot_track_id if present, else artist+title+chart_week
                merge_keys = ['mgphot_track_id'] if 'mgphot_track_id' in key_cols else key_cols
                # Find matching row
                cond = (aligned[merge_keys] == pd.Series({k: item.get(k) for k in merge_keys})).all(axis=1)
                if cond.any():
                    r = aligned[cond].iloc[0]
                    artist_qid = r.get('wd_artist_qid') if 'wd_artist_qid' in aligned.columns else None
                    work_qid = r.get('wd_work_qid') if 'wd_work_qid' in aligned.columns else None
                    wd_context = await wd_get_props(artist_qid, work_qid)
        except Exception:
            wd_context = {}

    return templates.TemplateResponse("index.html", {
        "request": request,
        "item": item,
        "genes": gene_rows,
        "neighbors": neighbors,
        "i": item['__index__'],
        "count": len(ITEMS),
        "wd": wd_context,
    })
