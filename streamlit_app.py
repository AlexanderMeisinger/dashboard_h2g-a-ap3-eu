#ToDo: Verweis auf Fabian Neumann; Button: Deutschland/EU + Button (2,3 Temperaturniveaus einstellen)

import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
import yaml
import plotly.graph_objects as go
from matplotlib.colors import to_rgba
from contextlib import suppress
import geopandas as gpd
import networkx as nx
import hvplot.networkx as hvnx
import holoviews as hv
import datetime
import hvplot.pandas
import plotly.express as px

from helpers import rename_techs_energy_balance, prepare_colors, rename_techs_h2_balance, rename_techs_capacities

CACHE_TTL = 24*3600 # seconds

#def plot_sankey(connections):

#    labels = np.unique(connections[["source", "target"]])

#    nodes = pd.Series({v: i for i, v in enumerate(labels)})

#    node_colors = pd.Series(nodes.index.map(colors).fillna("grey"), index=nodes.index)

#    link_colors = [
#        "rgba{}".format(to_rgba(node_colors[src], alpha=0.5))
#        for src in connections.source
#    ]

#    fig = go.Figure(
#        go.Sankey(
#            arrangement="snap",  # [snap, nodepad, perpendicular, fixed]
#            valuesuffix=" TWh",
#            valueformat=".1f",
#            node=dict(pad=4, thickness=10, label=nodes.index, color=node_colors),
#            link=dict(
#                source=connections.source.map(nodes),
#                target=connections.target.map(nodes),
#                value=connections.value,
#                label=connections.label,
#                color=link_colors,
#            ),
#        )
#    )

#    fig.update_layout(
#        height=800,
#        margin=dict(l=0, r=20, t=0, b=0)
#    )

#    return fig


#def plot_carbon_sankey(co2):

#    labels = np.unique(co2[["source", "target"]])

#    nodes = pd.Series({v: i for i, v in enumerate(labels)})

#    node_colors = pd.Series(nodes.index.map(colors).fillna("grey"), index=nodes.index)

#    link_colors = [
#        "rgba{}".format(to_rgba(colors[src], alpha=0.5))
#        for src in co2.label
#    ]

#    fig = go.Figure(
#        go.Sankey(
#            arrangement="freeform",  # [snap, nodepad, perpendicular, fixed]
#            valuesuffix=" MtCO2",
#            valueformat=".1f",
#            node=dict(
#                pad=4,
#                thickness=10,
#                label=nodes.index,
#                color=node_colors
#            ),
#            link=dict(
#                source=co2.source.map(nodes),
#                target=co2.target.map(nodes),
#                value=co2.value,
#                label=co2.label,
#                color=link_colors
#            ),
#        )
#    )

#    fig.update_layout(
#        height=800, 
#        margin=dict(l=100, r=0, t=0, b=150)
#    )

#    return fig


#@st.cache_data(ttl=CACHE_TTL)
#def nodal_balance(carrier, **kwargs):

#    ds = xr.open_dataset("data/time-series.nc")

#    df = ds[carrier].sel(**sel, drop=True).to_pandas().dropna(how='all', axis=1)

#    df = df.groupby(df.columns.map(rename_techs_tyndp), axis=1).sum()

#    df = df.loc[:, ~df.columns.isin(["H2 pipeline", "transmission lines"])]

#    missing = df.columns.difference(preferred_order)
#    order = preferred_order.intersection(df.columns).append(missing)
#    df = df.loc[:, order]
    
#    return df

#@st.cache_data(ttl=CACHE_TTL)
#def load_report(**kwargs):
#    ds1 = xr.open_dataset("data/resources.nc")
#    ds2 = xr.open_dataset("data/report.nc")
#    ds = xr.merge([ds1,ds2])
#    df = ds.sel(**sel, drop=True).to_dataframe().unstack(level=1).dropna(how='all', axis=1)

#    translate_0 = {
#        "demand": "Demand (TWh)",
#        "capacity_factor": "Capacity Factors (%)",
#        "cop": "Coefficient of Performance (-)",
#        "biomass_potentials": "Potentiacl (TWh)",
#        "salt_caverns": "Potential (TWh)",
#        "potential_used": "Used Potential (%)",
#        "curtailment": "Curtailment (%)",
#        "capacity": "Capacity (GW)",
#        "io": "Import-Export Balance (TWh)",
#        "lcoe": "Levelised Cost of Electricity (€/MWh)",
#        "market_value": "Market Values (€/MWh)",
#        "prices": "Market Prices (€/MWh)",
#        "storage": "Storage Capacity (GWh)",
#    }

#    translate_1 = {
#        "electricity": "Electricity",
#        "AC": "Electricity",
#        "transmission lines": "Electricity",
#        "H2": "Hydrogen",
#        "H2 storage": "hydrogen",
#        "hydrogen": "Hydrogen",
#        "oil": "Liquid Hydrocarbons",
#        "total": "Total",
#        "gas": "Methane",
#        "heat": "Heat",
#        "offwind-ac": "Offshore Wind (AC)",
#        "offwind-dc": "Offshore Wind (DC)",
#        "onwind": "Onshore Wind",
#        "onshore wind": "Onshore Wind",
#        "offshore wind": "Offshore Wind",
#        "hydroelectricity": "Hydro Electricity",
#        "PHS": "Pumped-hydro storage",
#        "hydro": "Hydro Reservoir",
#        "ror": "Run of River",
#        "solar": "Solar PV (utility)",
#        "solar PV": "Solar PV (utility)",
#        "solar rooftop": "Solar PV (rooftop)",
#        "ground heat pump": "Ground-sourced Heat Pump",
#        "air heat pump": "Air-sourced Heat Pump",
#        "biogas": "Biogas",
#        "biomass": "Biomass",
#        "solid biomass": "Solid Biomass",
#        "nearshore": "Hydrogen Storage (nearshore cavern)",
#        "onshore": "Hydrogen Storage (onshore cavern)",
#        "offshore": "Hydrogen Storage (offshore cavern)",
#    }

#    df.rename(columns=translate_0, level=0, inplace=True)
#    df.rename(columns=translate_1, level=1, inplace=True)

#    return df


#@st.cache_data(ttl=CACHE_TTL)
#def load_regions():
#    fn = "data/regions_onshore_elec_s_181.geojson"
#    gdf = gpd.read_file(fn).set_index('name')
#    gdf['name'] = gdf.index
#    gdf.geometry = gdf.to_crs(3035).geometry.simplify(1000).to_crs(4326)
#    return gdf


#@st.cache_data(ttl=CACHE_TTL)
#def load_positions():
#    buses = pd.read_csv("data/buses.csv", index_col=0)
#    return pd.concat([buses.x, buses.y], axis=1).apply(tuple, axis=1).to_dict()


#@st.cache_data(ttl=CACHE_TTL)
#def make_electricity_graph(**kwargs):

#    ds = xr.open_dataset("data/electricity-network.nc")
#    edges = ds.sel(**kwargs, drop=True).to_pandas()

#    edges["Total Capacity (GW)"] = edges.s_nom_opt.clip(lower=1e-3)
#    edges["Reinforcement (GW)"] = (edges.s_nom_opt - edges.s_nom).clip(lower=1e-3)
#    edges["Original Capacity (GW)"] = edges.s_nom.clip(lower=1e-3)
#    edges["Maximum Capacity (GW)"] = edges.s_nom_max.clip(lower=1e-3)
#    edges["Technology"] = edges.carrier
#    edges["Length (km)"] = edges.length

#    attr = ["Total Capacity (GW)", "Reinforcement (GW)", "Original Capacity (GW)", "Maximum Capacity (GW)", "Technology", "Length (km)"]
#    G = nx.from_pandas_edgelist(edges, 'bus0', 'bus1', edge_attr=attr)

#    return G

#@st.cache_data(ttl=CACHE_TTL)
#def make_hydrogen_graph(**kwargs):

#    ds = xr.open_dataset("data/hydrogen-network.nc")
#    edges = ds.sel(**kwargs, drop=True).to_pandas()

#    edges["Total Capacity (GW)"] = edges.p_nom_opt.clip(lower=1e-3)
#    edges["New Capacity (GW)"] = edges.p_nom_opt_new.clip(lower=1e-3)
#    edges["Retrofitted Capacity (GW)"] = edges.p_nom_opt_retro.clip(lower=1e-3)
#    edges["Maximum Retrofitting (GW)"] = edges.max_retro.clip(lower=1e-3)
#    edges["Length (km)"] = edges.length
#    edges["Name"] = edges.index

#    attr = ["Total Capacity (GW)", "New Capacity (GW)", "Retrofitted Capacity (GW)", "Maximum Retrofitting (GW)", "Length (km)", "Name"]
#    G = nx.from_pandas_edgelist(edges, 'bus0', 'bus1', edge_attr=attr)

#    return G


#def parse_spatial_options(x):
#    return " - ".join(x) if x != 'Nothing' else 'Nothing'


#@st.cache_data(ttl=CACHE_TTL)
#def load_summary(which):

#    df = pd.read_csv(f"data/{which}.csv", header=[0,1], index_col=0)

#   column_dict = {
#        "1.0": "without power expansion",
#        "opt": "with power grid expansion",
#        "H2 grid": "with hydrogen network",
#        "no H2 grid": "without hydrogen network",
#    }

#    df.rename(columns=column_dict, inplace=True)
#    df.columns = ["\n".join(col).strip() for col in df.columns.values]

#    df = df.groupby(df.index.map(rename_techs_tyndp), axis=0).sum()

#    missing = df.index.difference(preferred_order)
#    order = preferred_order.intersection(df.index).append(missing)
#    df = df.loc[order, :]

#    to_drop = df.index[df.abs().max(axis=1).fillna(0.0) < 1]
#    df.drop(to_drop, inplace=True)

#    return df[df.sum().sort_values().index].T


### MAIN

with open("data/config.yaml", encoding='utf-8') as file:
    config = yaml.safe_load(file)

preferred_order_energy_balance = pd.Index(config['preferred_order_energy_balance'])

## DISPLAY

st.set_page_config(
    page_title='H2Global meets Africa',
    layout="wide"
)

style = '<style>div.block-container{padding-top:.5rem; padding-bottom:0rem; padding-right:1.2rem; padding-left:1.2rem}</style>'
st.write(style, unsafe_allow_html=True)

## SIDEBAR

with st.sidebar:
    st.title("H2Global meets Africa: Energy demand modelling in Germany and the EU")

    st.markdown("""
        **FENES**
    """)
    # Explore trade-offs between power grid and hydrogen network expansion.


    pages = [
        "Europe",
        "Germany",
        #"Spatial configurations",
        #"System operation",
        #"Sankey of energy flows",
        #"Sankey of carbon flows"
    ]
    display = st.selectbox("Pages", pages, help="Choose your view on the system.")

    sel = {}

    #choices = {1: "yes", 0: "no"}
    #sel["power_grid"] = st.radio(
    #    ":zap: Electricity network expansion",
    #    choices, 
    #    format_func=lambda x: choices[x],
    #    horizontal=True
    #)

    #choices = {1: "yes", 0: "no"}
    #sel["hydrogen_grid"] = st.radio(
    #    ":droplet: Hydrogen network expansion",
    #    choices,
    #    format_func=lambda x: choices[x],
    #    horizontal=True
    #)

    #st.write("---")

    choices = {0: "2 °C", 1: "1.5 °C"}
    sel["low_carbon"] = st.radio(
        ":stopwatch: Temperature rise",
        choices,
        format_func=lambda x: choices[x],
        horizontal=True,
        help='Left button must be selected for all other choices in this segment.',
    )

    choices = {0: "yes", 1: "No"}
    sel["no_h2grid"] = st.radio(
        ":droplet: Hydrogen network",
        choices,
        format_func=lambda x: choices[x],
        horizontal=True,
        help='Left button must be selected for all other choices in this segment.',
    )
    # ToDo: Change icon
    choices = {0: "no", 1: "yes"}
    sel["ammonia"] = st.radio(
        ":earth_africa: Ammonia demand",
        choices,
        format_func=lambda x: choices[x],
        horizontal=True,
        help='Left button must be selected for all other choices in this segment.',
    )
    # ToDo: Change icon
    choices = {0: "no", 1: "yes"}
    sel["decentral"] = st.radio(
        ":wind_blowing_face: Decentral energy system",
        choices,
        format_func=lambda x: choices[x],
        horizontal=True,
        help='Left button must be selected for all other choices in this segment.',
    )

    number_sensitivities = sel["low_carbon"] + sel["no_h2grid"] + sel["ammonia"] + sel["decentral"]

    with st.expander("Details"):
         st.write("""
             All results were created using the open European energy system model
             PyPSA-Eur-Sec. The model covers all energy sectors including
             electricity, buildings, transport, agriculture and industry at high
             spatio-temporal resolution. The model code is available on
             [Github](http://github.com/pypsa/pypsa-eur-sec).
             """)

## PAGES

if (display == "Europe") and (number_sensitivities <= 1):

    st.title("Europe")

    choices = config["EU_scenarios"]
    idx = st.selectbox("View", choices, format_func=lambda x: choices[x], label_visibility='hidden')

    ds = xr.open_dataset("data/EU_scenarios_streamlit.nc")

    accessors = {k: v for k, v in sel.items() if k not in ['power_grid', 'hydrogen_grid']}
    df = ds[idx].sel(**accessors, drop=True).to_dataframe().squeeze().unstack(level=0).dropna(axis=1)

    df.index = ["".join(str(col)).strip() for col in df.index.values]

    if idx == "energy":
        df.columns = df.columns.map(rename_techs_energy_balance)
        df = df.groupby(axis=1, level=0).sum()
        to_drop = df.columns[(df.abs() < 50).all(axis=0)] # ToDo: Outsource energy threshold
        df.drop(columns=to_drop, inplace=True)
        missing = df.columns.difference(preferred_order_energy_balance)
        order = preferred_order_energy_balance.intersection(df.columns).append(missing)
        df = df.loc[:, order]
    elif idx == "hydrogen": 
        df.columns = df.columns.map(rename_techs_h2_balance)
        df = df.groupby(axis=1, level=0).sum()
        to_drop = df.columns[(df.abs() < 50).all(axis=0)] # ToDo: Outsource energy threshold
        df.drop(columns=to_drop, inplace=True)
    elif idx == "costs" or idx == "co2":
        df.columns = df.columns.map(rename_techs_energy_balance)
        df = df.groupby(axis=1, level=0).sum()
        to_drop = df.columns[(df.abs() < 1).all(axis=0)] # ToDo: Outsource energy threshold
        df.drop(columns=to_drop, inplace=True)
        missing = df.columns.difference(preferred_order_energy_balance)
        order = preferred_order_energy_balance.intersection(df.columns).append(missing)
        df = df.loc[:, order]
    elif idx == idx == "generation" or idx == "storage" or idx == "conversion":
        df.columns = df.columns.map(rename_techs_capacities)
        df = df.groupby(axis=1, level=0).sum()

    colors = prepare_colors(config)
    color = [colors[c] for c in df.columns]

    unit = choices[idx].split(" (")[1][:-1] # ugly

    ylim = config["ylim"][idx]

    plot = px.bar(
    df,
    x=df.index,  # Assuming the DataFrame index represents the x-axis
    y=df.columns,  # Stack the bars using the columns of the DataFrame
    color_discrete_sequence=color,  # Apply the color sequence
    labels={"value": f"{choices[idx]}", "index": ""},
    height=720,
    )

    # Update layout for font scaling and legend
    plot.update_layout(
        font=dict(size=18),  # Global font size, analogous to hvplot's fontscale
        xaxis=dict(
            title=dict(font=dict(size=18)),  # X-axis title font size
            tickfont=dict(size=16),  # X-axis tick labels font size
        ),
        yaxis=dict(
            title=dict(font=dict(size=18)),  # Y-axis title font size
            tickfont=dict(size=16),  # Y-axis tick labels font size
            tickformat=".0f"
        ),
        legend=dict(
            title=dict(text=""),  # Remove the legend title
            font=dict(size=16),  # Legend font size
        ),
    )

    # Add hover tooltips
    plot.update_traces(
        hovertemplate="Technology: %{x}<br>Value: %{y:.2f}<br>"
    )

    # Display the Plotly chart in Streamlit
    st.plotly_chart(plot, use_container_width=True)

if (display == "Germany") and (number_sensitivities <= 1):

    st.title("Germany")

    choices = config["DE_scenarios"]
    idx = st.selectbox("View", choices, format_func=lambda x: choices[x], label_visibility='hidden')

    ds = xr.open_dataset("data/DE_scenarios_streamlit.nc")

    accessors = {k: v for k, v in sel.items() if k not in ['power_grid', 'hydrogen_grid']}
    df = ds[idx].sel(**accessors, drop=True).to_dataframe().squeeze().unstack(level=0).dropna(axis=1)

    df.index = ["".join(str(col)).strip() for col in df.index.values]

    if idx == "energy":
        df.columns = df.columns.map(rename_techs_energy_balance)
        df = df.groupby(axis=1, level=0).sum()
        to_drop = df.columns[(df.abs() < 50).all(axis=0)] # ToDo: Outsource energy threshold
        df.drop(columns=to_drop, inplace=True)
        missing = df.columns.difference(preferred_order_energy_balance)
        order = preferred_order_energy_balance.intersection(df.columns).append(missing)
        df = df.loc[:, order]
    elif idx == "hydrogen": 
        df.columns = df.columns.map(rename_techs_h2_balance)
        df = df.groupby(axis=1, level=0).sum()
        to_drop = df.columns[(df.abs() < 50).all(axis=0)] # ToDo: Outsource energy threshold
        df.drop(columns=to_drop, inplace=True)
    elif idx == "costs" or idx == "co2":
        df.columns = df.columns.map(rename_techs_energy_balance)
        df = df.groupby(axis=1, level=0).sum()
        to_drop = df.columns[(df.abs() < 1).all(axis=0)] # ToDo: Outsource energy threshold
        df.drop(columns=to_drop, inplace=True)
        missing = df.columns.difference(preferred_order_energy_balance)
        order = preferred_order_energy_balance.intersection(df.columns).append(missing)
        df = df.loc[:, order]
    elif idx == idx == "generation" or idx == "storage" or idx == "conversion":
        df.columns = df.columns.map(rename_techs_capacities)
        df = df.groupby(axis=1, level=0).sum()

    colors = prepare_colors(config)
    color = [colors[c] for c in df.columns]

    unit = choices[idx].split(" (")[1][:-1] # ugly

    ylim = config["ylim"][idx]

    plot = px.bar(
    df,
    x=df.index,  # Assuming the DataFrame index represents the x-axis
    y=df.columns,  # Stack the bars using the columns of the DataFrame
    color_discrete_sequence=color,  # Apply the color sequence
    labels={"value": f"{choices[idx]}", "index": ""},
    height=720,
    )

    # Update layout for font scaling and legend
    plot.update_layout(
        font=dict(size=18),  # Global font size, analogous to hvplot's fontscale
        xaxis=dict(
            title=dict(font=dict(size=18)),  # X-axis title font size
            tickfont=dict(size=16),  # X-axis tick labels font size
        ),
        yaxis=dict(
            title=dict(font=dict(size=18)),  # Y-axis title font size
            tickfont=dict(size=16),  # Y-axis tick labels font size
            tickformat=".0f"
        ),
        legend=dict(
            title=dict(text=""),  # Remove the legend title
            font=dict(size=16),  # Legend font size
        ),
    )

    # Add hover tooltips
    plot.update_traces(
        hovertemplate="Technology: %{x}<br>Value: %{y:.2f}<br>"
    )

    # Display the Plotly chart in Streamlit
    st.plotly_chart(plot, use_container_width=True)


if number_sensitivities > 1:
    
    st.write("")
    st.write("")

    message = "Sorry, you can only choose one additional sensitivity in the lower block of the left panel!"
    st.error(message, icon="🚨")