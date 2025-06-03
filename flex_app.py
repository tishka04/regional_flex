import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

SCENARIOS = {
    "Full year": "full_year.csv",
    "Winter weekday": "winter_weekday.csv",
}

REGIONS = [
    "Auvergne_Rhone_Alpes",
    "Nouvelle_Aquitaine",
    "Occitanie",
    "Provence_Alpes_Cote_dAzur",
]

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

st.sidebar.title("Regional Flex Explorer")
scenario = st.sidebar.selectbox("Scénario", list(SCENARIOS.keys()))

# Load data for the selected scenario
csv_path = SCENARIOS[scenario]
if not Path(csv_path).exists():
    st.error(f"File {csv_path} not found")
    st.stop()

df = load_data(csv_path)

region = st.sidebar.selectbox("Région", REGIONS)

min_date = df.index.min().to_pydatetime()
max_date = df.index.max().to_pydatetime()
date_range = st.sidebar.slider(
    "Période",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD",
)

# Technologies available for dispatch
tech_options = [
    col.replace(f"dispatch_", "").replace(f"_{region}", "")
    for col in df.columns
    if col.startswith("dispatch_") and col.endswith(region)
]
selected_techs = st.sidebar.multiselect(
    "Technologies (dispatch)", tech_options, default=tech_options
)

start, end = date_range
dff = df.loc[start:end].copy()

# Dispatch plot
dispatch_cols = [f"dispatch_{tech}_{region}" for tech in selected_techs]
if dispatch_cols:
    fig = px.area(
        dff,
        y=dispatch_cols,
        labels={"value": "MW"},
        title=f"Mix de production - {region}",
    )
    st.plotly_chart(fig, use_container_width=True)

# Storage state of charge and power
for storage in ["STEP", "batteries"]:
    soc_col = f"storage_soc_{storage}_{region}"
    if soc_col in dff.columns:
        fig_soc = px.line(dff, y=[soc_col], title=f"État de charge {storage} - {region}")
        st.plotly_chart(fig_soc, use_container_width=True)
    pow_cols = [
        c
        for c in [f"storage_charge_{storage}_{region}", f"storage_discharge_{storage}_{region}"]
        if c in dff.columns
    ]
    if pow_cols:
        fig_pow = px.line(dff, y=pow_cols, title=f"Puissance {storage} - {region}")
        st.plotly_chart(fig_pow, use_container_width=True)

# Demand response
cols_dr = [c for c in [f"demand_response_{region}", f"dr_active_{region}"] if c in dff.columns]
if cols_dr:
    fig_dr = px.line(dff, y=cols_dr, title=f"Demand Response - {region}")
    st.plotly_chart(fig_dr, use_container_width=True)

# Slack values
slack_cols = [c for c in [f"slack_pos_{region}", f"slack_neg_{region}"] if c in dff.columns]
if slack_cols:
    fig_slack = px.line(dff, y=slack_cols, title=f"Slack ± - {region}")
    st.plotly_chart(fig_slack, use_container_width=True)

# Nodal price
price_col = f"nodal_price_{region}"
if price_col in dff.columns:
    fig_price = px.line(dff, y=[price_col], title=f"Prix nodal - {region}")
    st.plotly_chart(fig_price, use_container_width=True)

# Imports/Exports
out_cols = [c for c in dff.columns if c.startswith(f"flow_out_{region}_")]
in_cols = [c for c in dff.columns if c.startswith("flow_out_") and c.endswith(f"_{region}")]
if out_cols or in_cols:
    dff["exports"] = dff[out_cols].sum(axis=1) if out_cols else 0
    dff["imports"] = dff[in_cols].sum(axis=1) if in_cols else 0
    dff["net_flow"] = dff["imports"] - dff["exports"]
    fig_flows = px.line(dff, y=["imports", "exports", "net_flow"], title=f"Flux - {region}")
    st.plotly_chart(fig_flows, use_container_width=True)

