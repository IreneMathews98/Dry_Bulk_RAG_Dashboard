"""Spot timecharter averages and FFA forward prices for the opening page.

Built the same way as FFA_spot.ipynb (Baltic Forward Assessments for the 5TC,
4TC and 10TC contracts next to the C5TC / P4TC / S10TC spot averages), except
the merge is outer so FFA dates after the last spot date are kept too.
Loaded once and cached; returns None if the data files aren't there.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

SEGMENTS = [
    {"key": "cape", "name": "Capesize", "spot": "C5TC", "ffa": "5TC_C", "file": "bfa_cape.csv"},
    {"key": "panamax", "name": "Panamax", "spot": "P4TC", "ffa": "4TC_P", "file": "bfa_panamax.csv"},
    {"key": "supramax", "name": "Supramax", "spot": "S10TC", "ffa": "10TC_S", "file": "bfa_supramax.csv"},
]
CONTRACTS = [("CURMON", "FFA current month"), ("+1MON", "FFA +1 month"), ("+2MON", "FFA +2 months")]

_cache: dict | None = None


def build_prices() -> dict | None:
    freight_path = DATA_DIR / "Freight data.csv"
    if not freight_path.exists() or not all((DATA_DIR / s["file"]).exists() for s in SEGMENTS):
        return None

    freight = pd.read_csv(freight_path, usecols=["Date"] + [s["spot"] for s in SEGMENTS])
    freight["Date"] = pd.to_datetime(freight["Date"], format="%m-%d-%y")
    table = freight.set_index("Date")

    for seg in SEGMENTS:
        ffa_ids = [seg["ffa"] + c for c, _ in CONTRACTS]
        bfa = pd.read_csv(DATA_DIR / seg["file"], usecols=["ArchiveDate", "RouteIdentifier", "RouteAverage"])
        bfa = bfa[bfa["RouteIdentifier"].isin(ffa_ids)]
        bfa["Date"] = pd.to_datetime(bfa["ArchiveDate"], format="%d-%b-%Y")
        wide = bfa.pivot_table(index="Date", columns="RouteIdentifier", values="RouteAverage", aggfunc="last")
        table = table.join(wide, how="outer")

    table = table.sort_index()
    segments = []
    for seg in SEGMENTS:
        series = [{"id": seg["spot"], "label": f"Spot {seg['spot']}"}]
        series += [{"id": seg["ffa"] + c, "label": label} for c, label in CONTRACTS]
        segments.append({"key": seg["key"], "name": seg["name"], "series": series})

    values = {}
    for col in table.columns:
        s = table[col]
        values[col] = [None if pd.isna(v) else round(float(v)) for v in s]
    return {
        "dates": table.index.strftime("%Y-%m-%d").tolist(),
        "values": values,
        "segments": segments,
        "first": table.index.min().strftime("%d %b %Y"),
        "last": table.index.max().strftime("%d %b %Y"),
    }


def get_prices() -> dict | None:
    global _cache
    if _cache is None:
        _cache = build_prices()
    return _cache
