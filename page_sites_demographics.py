import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
from data_prep import load_and_preprocess_brca

def page_demographics():
    """
    Global and demographic patterns of breast cancer subtypes.

    Features:
    - Subtype filter with normalization toggle
    - Age range slider
    - World map showing total BRCA incidence by country
    - Subtype composition bar chart linked to map
    """
    st.header("Global and Demographic Patterns of Breast Cancer Subtypes")
    
    # Load data
    try:
        brca_df = load_and_preprocess_brca("clinical.tsv")
    except:
        st.error("Error: Could not locate clinical.tsv")
        return
    

    st.sidebar.subheader("Filters")
    
    # Subtype filter
    subtype_opts = sorted([x for x in brca_df["stage"].unique() if pd.notna(x) and x != "Unknown"])
    selected_subtype = st.sidebar.selectbox(
        "SUBTYPE",
        ["All Subtypes"] + subtype_opts
    )
    
    # Normalize by toggle
    normalize_by = st.sidebar.radio(
        "Normalize by",
        ["Total Cases", "Absolute Counts"],
        horizontal=False
    )
    
    # Filter text box
    filter_text = st.sidebar.text_input("Filter", "")
    
    # Age range slider
    age_range = st.sidebar.slider(
        "Age Range",
        int(brca_df["age"].min()),
        int(brca_df["age"].max()),
        (int(brca_df["age"].min()), int(brca_df["age"].max()))
    )
    
    # Include unknown/missing checkbox
    include_unknown = st.sidebar.checkbox("Include unknown / missing", value=True)
    
    # --- Apply Filters ---
    filtered_df = brca_df.copy()
    
    # Subtype filter
    if selected_subtype != "All Subtypes":
        filtered_df = filtered_df[filtered_df["stage"] == selected_subtype]
    
    # Age range filter
    filtered_df = filtered_df[(filtered_df["age"] >= age_range[0]) & (filtered_df["age"] <= age_range[1])]
    
    # Unknown/missing filter
    if not include_unknown:
        filtered_df = filtered_df[filtered_df["stage"] != "Unknown"]
    
    # Text filter (searches across race, ethnicity, site)
    if filter_text:
        mask = (
            filtered_df["race"].str.contains(filter_text, case=False, na=False) |
            filtered_df["ethnicity"].str.contains(filter_text, case=False, na=False) |
            filtered_df["site"].str.contains(filter_text, case=False, na=False)
        )
        filtered_df = filtered_df[mask]
    
    if filtered_df.empty:
        st.warning("No data matches current filters. Please adjust your selection.")
        return
    
    st.sidebar.metric("Patients in View", len(filtered_df))
    
    # Geographic Incidence Map ---
    st.subheader("World Map: Total BRCA Incidence by Country")
    
    # Aggregate real country data from clinical.tsv
    country_counts = (
        filtered_df["country"]
        .dropna()
        .replace("Unknown", pd.NA)
        .dropna()
        .value_counts()
        .reset_index()
    )
    country_counts.columns = ["Country", "Cases"]

    if not country_counts.empty:
        fig = px.choropleth(
            country_counts,
            locations="Country",
            locationmode="country names",
            color="Cases",
            hover_name="Country",
            color_continuous_scale="Blues",
            labels={"Cases": "BRCA Cases"},
        )
        fig.update_layout(height=500, font=dict(size=11))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📍 Hover over countries to see case counts (derived from clinical.tsv)")
    else:
        st.info("No country data available after applying filters.")
    
    # --- Subtype Composition ---
    st.subheader("Subtype Distribution")
    
    # Subtype distribution (using stage as proxy for subtype)
    subtype_counts = filtered_df["stage"].value_counts().reset_index()
    subtype_counts.columns = ["Subtype", "Count"]
    subtype_counts = subtype_counts[subtype_counts["Subtype"] != "Unknown"]
    
    # Define consistent color scheme for subtypes
    subtype_colors = {
        "Stage I": "#1f77b4",
        "Stage II": "#ff7f0e",
        "Stage III": "#2ca02c",
        "Stage IV": "#d62728"
    }
    
    # Subtype distribution by race/ethnicity
    subtype_by_demo = filtered_df.groupby(["stage", "race"]).size().reset_index(name="Count")
    subtype_by_demo = subtype_by_demo[subtype_by_demo["stage"] != "Unknown"]
    subtype_by_demo = subtype_by_demo[subtype_by_demo["race"] != "Unknown"]
    
    if not subtype_by_demo.empty:
        if normalize_by == "Total Cases":
            total = subtype_by_demo["Count"].sum()
            subtype_by_demo["Percentage"] = (100 * subtype_by_demo["Count"] / total).round(1)
            tooltip_fields = ["stage", "race", "Count", "Percentage"]
        else:
            tooltip_fields = ["stage", "race", "Count"]
        
        # Create treemap showing subtype distribution by race
        treemap_chart = alt.Chart(subtype_by_demo).mark_rect().encode(
            x=alt.X("race:N", title="Race"),
            y=alt.Y("stage:N", title="Stage/Subtype"),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), title="Cases"),
            size="Count:Q",
            tooltip=tooltip_fields
        ).properties(height=300, width=600, title="Subtype Distribution by Race/Ethnicity")
        
        st.altair_chart(treemap_chart, use_container_width=True)
    else:
        st.info("No subtype data available for current filters.")
    st.divider()
    
    # --- Stage at Diagnosis by Demographics ---
    st.subheader("Stage at Diagnosis by Demographics")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Stage Distribution by Race**")
        
        # Stage by race
        stage_by_race = filtered_df.groupby(["race", "stage"]).size().reset_index(name="Count")
        stage_by_race = stage_by_race[stage_by_race["race"] != "Unknown"]
        stage_by_race = stage_by_race[stage_by_race["stage"] != "Unknown"]
        
        if not stage_by_race.empty:
            # Calculate percentages within each race
            stage_by_race["Total"] = stage_by_race.groupby("race")["Count"].transform("sum")
            stage_by_race["Percentage"] = (100 * stage_by_race["Count"] / stage_by_race["Total"]).round(1)
            
            stage_race_chart = alt.Chart(stage_by_race).mark_bar().encode(
                x=alt.X("Percentage:Q", title="Percentage (%)", stack="normalize"),
                y=alt.Y("race:N", title="Race"),
                color=alt.Color("stage:N", title="Stage", scale=alt.Scale(scheme="category10")),
                tooltip=["race", "stage", "Count", "Percentage"]
            ).properties(height=300, title="Stage Distribution by Race (100% Stacked)")
            
            st.altair_chart(stage_race_chart, use_container_width=True)
        else:
            st.info("No stage/race data available.")
    
    with col2:
        st.write("**Stage Distribution by Ethnicity**")
        
        # Stage by ethnicity
        stage_by_ethnicity = filtered_df.groupby(["ethnicity", "stage"]).size().reset_index(name="Count")
        stage_by_ethnicity = stage_by_ethnicity[stage_by_ethnicity["ethnicity"] != "Unknown"]
        stage_by_ethnicity = stage_by_ethnicity[stage_by_ethnicity["stage"] != "Unknown"]
        
        if not stage_by_ethnicity.empty:
            # Calculate percentages within each ethnicity
            stage_by_ethnicity["Total"] = stage_by_ethnicity.groupby("ethnicity")["Count"].transform("sum")
            stage_by_ethnicity["Percentage"] = (100 * stage_by_ethnicity["Count"] / stage_by_ethnicity["Total"]).round(1)
            
            stage_ethnicity_chart = alt.Chart(stage_by_ethnicity).mark_bar().encode(
                x=alt.X("Percentage:Q", title="Percentage (%)", stack="normalize"),
                y=alt.Y("ethnicity:N", title="Ethnicity"),
                color=alt.Color("stage:N", title="Stage", scale=alt.Scale(scheme="category10")),
                tooltip=["ethnicity", "stage", "Count", "Percentage"]
            ).properties(height=300, title="Stage Distribution by Ethnicity (100% Stacked)")
            
            st.altair_chart(stage_ethnicity_chart, use_container_width=True)
        else:
            st.info("No stage/ethnicity data available.")
    
    st.divider()
    
    st.divider()
    
    # --- Summary Statistics ---
    st.subheader("Summary Statistics")
    
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.metric("Total Patients", len(filtered_df))
    
    with summary_col2:
        unique_sites = filtered_df["site"].nunique()
        st.metric("Unique Sites", unique_sites)
    
    with summary_col3:
        unique_stages = filtered_df[filtered_df["stage"] != "Unknown"]["stage"].nunique()
        st.metric("Stages Represented", unique_stages)
    
    with summary_col4:
        avg_age = filtered_df["age"].mean()
        st.metric("Avg Age at Diagnosis", f"{avg_age:.1f}")


if __name__ == "__main__":
    page_demographics()
