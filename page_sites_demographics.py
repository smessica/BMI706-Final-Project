import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
from data_prep import load_and_preprocess_brca

# --- Page B: Tumor Site + Demographics Map View ---

def page_sites_demographics():
    """
    Global and demographic patterns of breast cancer subtypes.
    
    Features:
    - Subtype filter with normalization toggle
    - Age range slider
    - World map showing total BRCA incidence by country
    - Subtype composition bar chart linked to map
    - Consistent color encodings across views
    """
    st.header("Global and Demographic Patterns of Breast Cancer Subtypes")
    
    # Load data
    try:
        brca_df = load_and_preprocess_brca("clinical.tsv")
    except:
        st.error("Error: Could not locate clinical.tsv")
        return
    
    # --- Sidebar Filters (matching sketch) ---
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
    
    # --- Lindsay's View: Geographic Incidence Map ---
    st.subheader("World Map: Total BRCA Incidence by Country")
    
    # Create sample country data for demonstration
    # In production, this would come from geocoding patient addresses
    country_data = pd.DataFrame({
        'Country': ['United States', 'Canada', 'United Kingdom', 'Germany', 'France', 
                   'Australia', 'Japan', 'Brazil', 'Mexico', 'India'],
        'Cases': [1200, 450, 380, 320, 290, 250, 200, 180, 150, 140],
        'ISO-3': ['USA', 'CAN', 'GBR', 'DEU', 'FRA', 'AUS', 'JPN', 'BRA', 'MEX', 'IND']
    })
    
    # Create choropleth map
    fig = px.choropleth(
        country_data,
        locations='ISO-3',
        color='Cases',
        hover_name='Country',
        color_continuous_scale='Blues',
        labels={'Cases': 'BRCA Cases'}
    )
    fig.update_layout(height=500, font=dict(size=11))
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("📍 Hover over countries to see case counts. (Note: Demo data shown; update with real country-level aggregation)")
    
    st.divider()
    
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
    
    if not subtype_counts.empty:
        if normalize_by == "Total Cases":
            subtype_counts["Percentage"] = 100 * subtype_counts["Count"] / subtype_counts["Count"].sum()
            encoding_field = "Percentage:Q"
            title_suffix = " (%)"
        else:
            encoding_field = "Count:Q"
            title_suffix = ""
        
        subtype_chart = alt.Chart(subtype_counts).mark_bar().encode(
            x=alt.X(encoding_field, title=f"Cases{title_suffix}"),
            y=alt.Y("Subtype:N", sort="-x", title="Stage/Subtype"),
            color=alt.Color("Subtype:N", scale=alt.Scale(domain=list(subtype_colors.keys()), 
                                                         range=list(subtype_colors.values())),
                           title="Subtype"),
            tooltip=["Subtype", "Count"]
        ).properties(height=300, width=400)
        
        st.altair_chart(subtype_chart, use_container_width=True)
    else:
        st.info("No subtype data available for current filters.")
    
    st.divider()
    
    # --- Lucy's View: Tumor Site & Co-occurrence ---
    st.subheader("Tumor Site Analysis")
    
    col_site1, col_site2 = st.columns([1.2, 2])
    
    with col_site1:
        st.write("**Primary Sites**")
        
        # Site distribution
        site_counts = filtered_df["site"].value_counts().reset_index()
        site_counts.columns = ["Site", "Count"]
        site_counts = site_counts[site_counts["Site"] != "Unknown"].head(10)
        
        if not site_counts.empty:
            site_chart = alt.Chart(site_counts).mark_bar(color="#4c78a8").encode(
                x=alt.X("Count:Q", title="Number of Patients"),
                y=alt.Y("Site:N", sort="-x", title="Anatomical Site"),
                tooltip=["Site", "Count"]
            ).properties(height=300)
            
            st.altair_chart(site_chart, use_container_width=True)
    
    with col_site2:
        st.write("**Demographics by Age**")
        age_hist = filtered_df[["age"]].dropna()
        
        if not age_hist.empty:
            age_chart = alt.Chart(age_hist).mark_bar(color="#72b7b2").encode(
                x=alt.X("age:Q", bin=alt.Bin(maxbins=15), title="Age at Diagnosis"),
                y=alt.Y("count():Q", title="Number of Patients"),
                tooltip=["count()"]
            ).properties(height=300, title="Age Distribution")
            
            st.altair_chart(age_chart, use_container_width=True)
        else:
            st.info("No age data available.")
    
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
    page_sites_demographics()
