#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualise interregional energy flows on a map and as a Sankey diagram."""
import argparse
import pickle
from pathlib import Path

import plotly.graph_objects as go
import folium

# Rough coordinates for the four French regions (latitude, longitude)
REGION_COORDS = {
    "Auvergne_Rhone_Alpes": (45.75, 4.85),
    "Nouvelle_Aquitaine": (44.84, -0.58),
    "Occitanie": (43.60, 3.88),
    "Provence_Alpes_Cote_dAzur": (43.30, 5.37),
}


def load_results(path: Path) -> dict:
    """Load a pickled results dictionary."""
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_net_flows(res: dict) -> dict:
    """Return net flows between region pairs as { (i,j): value }."""
    flows = {}
    regions = res.get("regions", list(REGION_COORDS))
    for i, r1 in enumerate(regions):
        for r2 in regions[i + 1 :]:
            k1 = f"flow_out_{r1}_{r2}"
            k2 = f"flow_out_{r2}_{r1}"
            v12 = sum(res["variables"].get(k1, {}).values())
            v21 = sum(res["variables"].get(k2, {}).values())
            flows[(r1, r2)] = v12 - v21
    return flows


def sankey_figure(flows: dict) -> go.Figure:
    regions = sorted({r for pair in flows for r in pair})
    idx = {r: i for i, r in enumerate(regions)}
    sources = []
    targets = []
    values = []
    for (r1, r2), val in flows.items():
        if val > 0:
            sources.append(idx[r1])
            targets.append(idx[r2])
            values.append(val)
        elif val < 0:
            sources.append(idx[r2])
            targets.append(idx[r1])
            values.append(-val)
    fig = go.Figure(
        go.Sankey(
            node=dict(label=regions),
            link=dict(source=sources, target=targets, value=values),
        )
    )
    fig.update_layout(title="Interregional energy exchanges")
    return fig


def folium_map(flows: dict) -> folium.Map:
    m = folium.Map(location=[46.2, 2.2], zoom_start=6, tiles="cartodbpositron")
    for region, (lat, lon) in REGION_COORDS.items():
        folium.CircleMarker(location=[lat, lon], radius=5, popup=region).add_to(m)
    scale = max(abs(v) for v in flows.values()) or 1
    for (r1, r2), val in flows.items():
        if val == 0:
            continue
        latlon1 = REGION_COORDS.get(r1)
        latlon2 = REGION_COORDS.get(r2)
        if not latlon1 or not latlon2:
            continue
        color = "blue" if val > 0 else "red"
        weight = max(1, 5 * abs(val) / scale)
        folium.PolyLine(locations=[latlon1, latlon2], color=color, weight=weight).add_to(m)
    return m


def main():
    pa = argparse.ArgumentParser(description="Map interregional flows")
    pa.add_argument("--pickle", required=True)
    pa.add_argument("--out", default="plots")
    args = pa.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = load_results(Path(args.pickle))
    flows = compute_net_flows(res)

    fig = sankey_figure(flows)
    fig.write_image(out_dir / "interregional_sankey.png")

    m = folium_map(flows)
    m.save(out_dir / "interregional_flows_map.html")
    print(f"Outputs saved in {out_dir}")


if __name__ == "__main__":
    main()
