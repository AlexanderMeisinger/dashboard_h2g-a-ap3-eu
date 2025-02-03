# This script visualize results from PyPSA-Eur via Streamlit
# Author: Alexander Meisinger
# Base: https://github.com/fneum/spatial-sector-dashboard and https://github.com/PyPSA/pypsa-eur

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

from helpers import rename_techs_energy_balance, prepare_colors, rename_techs_h2_balance, rename_tech_capacity

CACHE_TTL = 24*3600 # seconds

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

    pages = [
        "Europe",
        "Germany",
    ]
    display = st.selectbox("Pages", pages, help="Choose your view on the system.")

    sel = {}

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

    st.markdown("<br>", unsafe_allow_html=True)
    st.title("Europe")

    choices = config["EU_scenarios"]
    idx = st.selectbox("View", choices, format_func=lambda x: choices[x], label_visibility='hidden')

    ds = xr.open_dataset("data/EU_scenarios_streamlit_2.0-C.nc")

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
    elif idx == "storage" or idx == "generation" or idx == "conversion":
        df.columns = df.columns.map(rename_tech_capacity)
        df = df.groupby(axis=1, level=0).sum()
        
        to_drop = df.columns[(df.abs() < 1).all(axis=0)] # ToDo: Outsource energy threshold
        df.drop(columns=to_drop, inplace=True)
    else:
        df.columns = df.columns.map(rename_techs_energy_balance)
        df = df.groupby(axis=1, level=0).sum()
        
        to_drop = df.columns[(df.abs() < 1).all(axis=0)] # ToDo: Outsource energy threshold
        df.drop(columns=to_drop, inplace=True)
        
        missing = df.columns.difference(preferred_order_energy_balance)
        order = preferred_order_energy_balance.intersection(df.columns).append(missing)
        df = df.loc[:, order]


    #ToDo: Check storage
    if idx == 'storage':
         df.drop("co2", axis=1, inplace=True, errors="ignore")
         df.drop("co2 sequestered", axis=1, inplace=True, errors="ignore")
         df.drop("electricity distribution grid", axis=1, inplace=True, errors="ignore")
         df.drop("methanol", axis=1, inplace=True, errors="ignore")
         df.drop("oil", axis=1, inplace=True, errors="ignore")
         df.drop("oil refining", axis=1, inplace=True, errors="ignore")
         df.drop("solar rooftop", axis=1, inplace=True, errors="ignore")
         df.drop("solid biomass", axis=1, inplace=True, errors="ignore")
         df.drop("unsustainable biogas", axis=1, inplace=True, errors="ignore")
         df.drop("unsustainable bioliquids", axis=1, inplace=True, errors="ignore")
         df.drop("unsustainable solid biomass", axis=1, inplace=True, errors="ignore")
         df.drop("Solar", axis=1, inplace=True, errors="ignore")
         df.drop("biogas", axis=1, inplace=True, errors="ignore")
         df.drop("gas", axis=1, inplace=True, errors="ignore")
         df.drop("ammonia_store", axis=1, inplace=True, errors="ignore") # Check again

    # ToDo: Check biomass capacities
    if idx == 'generation':
        df.drop("biogas", axis=1, inplace=True, errors="ignore")
        df.drop("solid biomass", axis=1, inplace=True, errors="ignore")
        df.drop("unsustainable biogas", axis=1, inplace=True, errors="ignore")
        df.drop("unsustainable bioliquids", axis=1, inplace=True, errors="ignore")
        df.drop("unsustainable solid biomass", axis=1, inplace=True, errors="ignore")

    # ToDo: Check biomass capacities
    if idx == 'conversion':
        df.drop("unsustainable bioliquids", axis=1, inplace=True, errors="ignore")


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

    st.markdown("<br>", unsafe_allow_html=True)
    st.title("Germany")

    choices = config["DE_scenarios"]
    idx = st.selectbox("View", choices, format_func=lambda x: choices[x], label_visibility='hidden')

    ds = xr.open_dataset("data/DE_scenarios_streamlit_2.0-C.nc")

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
        
        to_drop = df.columns[(df.abs() < 1).all(axis=0)] # ToDo: Outsource energy threshold
        df.drop(columns=to_drop, inplace=True)
    elif idx == "storage" or idx == "generation" or idx == "conversion":
        df.columns = df.columns.map(rename_tech_capacity)
        df = df.groupby(axis=1, level=0).sum()
        
        to_drop = df.columns[(df.abs() < 1).all(axis=0)] # ToDo: Outsource energy threshold
        df.drop(columns=to_drop, inplace=True)
    else:
        df.columns = df.columns.map(rename_techs_energy_balance)
        df = df.groupby(axis=1, level=0).sum()
        
        to_drop = df.columns[(df.abs() < 1).all(axis=0)] # ToDo: Outsource energy threshold
        df.drop(columns=to_drop, inplace=True)
        
        missing = df.columns.difference(preferred_order_energy_balance)
        order = preferred_order_energy_balance.intersection(df.columns).append(missing)
        df = df.loc[:, order]

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