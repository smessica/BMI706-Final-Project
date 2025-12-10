import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
from lifelines import KaplanMeierFitter

# --- Page Config ---
st.set_page_config(page_title="BRCA Survival Analysis", layout="wide")

# --- Utility & Data Loading ---
BAD_TOKENS = {"'--", "--", "NA", "Na", "N/A", "Not Reported", "Unknown", "", "nan", "NaN"}

def clean_string_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace(list(BAD_TOKENS), np.nan)

def clean_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).replace(list(BAD_TOKENS), np.nan), errors="coerce")

@st.cache_data
def load_and_preprocess_brca(path: str):
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    
    # Isolate Breast cancer cohort
    df["tumor_type"] = clean_string_series(df.get("diagnoses.tissue_or_organ_of_origin", pd.Series("Other")))
    brca_df = df[df["tumor_type"].str.contains("Breast", case=False, na=False)].copy()
    
    # Survival & Event status
    if "demographic.vital_status" in brca_df.columns:
        brca_df["event"] = brca_df["demographic.vital_status"].map({"Dead": 1, "Alive": 0}).fillna(0).astype(int)
    
    death = clean_numeric_series(brca_df.get("demographic.days_to_death", pd.Series(np.nan)))
    follow = clean_numeric_series(brca_df.get("diagnoses.days_to_last_follow_up", pd.Series(np.nan)))
    brca_df["time"] = np.where(death.notna(), death, follow)
    
    # Clinical variables
    brca_df["age"] = clean_numeric_series(brca_df.get("demographic.age_at_index", pd.Series(np.nan)))
    brca_df["stage"] = clean_string_series(brca_df.get("diagnoses.ajcc_pathologic_stage", pd.Series("Unknown")))
    brca_df["treatment"] = clean_string_series(brca_df.get("treatments.treatment_type", pd.Series("Other")))
    brca_df["agent"] = clean_string_series(brca_df.get("treatments.therapeutic_agents", pd.Series("Not Specified")))
    
    return brca_df.dropna(subset=["time"])

# --- Visualizations ---
def plot_km_brca(df, therapy_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    kmf = KaplanMeierFitter()
    
    valid_agents = df[df["agent"] != "Not Specified"]
    if not valid_agents.empty and valid_agents["agent"].nunique() > 1:
        top_labels = valid_agents["agent"].value_counts().nlargest(5).index
        for label in top_labels:
            group = df[df["agent"] == label]
            if len(group) >= 2:
                kmf.fit(group["time"], group["event"], label=str(label))
                kmf.plot_survival_function(ax=ax, ci_show=False)
    else:
        # Pass the dynamic therapy name here to replace the static "Selected Cohort"
        kmf.fit(df["time"], df["event"], label=f"{therapy_name} Cohort")
        kmf.plot_survival_function(ax=ax)
            
    ax.set_title(f"Survival Probability: {therapy_name}")
    ax.set_ylabel("Survival Probability")
    ax.set_xlabel("Days since Diagnosis")
    plt.tight_layout(pad=2.0)
    return fig

# --- Main App Logic ---
def main():
    st.title("BRCA Clinical Survival Analysis Dashboard")
    
    try:
        brca_df = load_and_preprocess_brca("clinical.tsv")
    except:
        st.error("Error: Could not locate clinical.tsv in the specified directory.")
        return

    # --- 1. Population Selectors ---
    st.write("##### 1. Define Population filters")
    f1, f2 = st.columns(2)
    with f1:
        age_label = st.selectbox("Define Age Range:", ["All BRCA", "Young (0-45)", "Adult (45-65)", "Senior (65+)"])
    with f2:
        stage_opts = sorted([str(x) for x in brca_df["stage"].unique() if pd.notna(x)])
        selected_stage = st.selectbox("Define Cancer Stage:", ["All Stages"] + stage_opts)

    # Global filter processing
    filtered_df = brca_df.copy()
    if selected_stage != "All Stages": 
        filtered_df = filtered_df[filtered_df["stage"] == selected_stage]
    
    if "Young" in age_label: filtered_df = filtered_df[filtered_df["age"] <= 45]
    elif "Adult" in age_label: filtered_df = filtered_df[(filtered_df["age"] > 45) & (filtered_df["age"] <= 65)]
    elif "Senior" in age_label: filtered_df = filtered_df[filtered_df["age"] > 65]

    # --- 2. Cohort Demographic Distributions ---
    st.write("##### 2. Cohort Demographics (Filtered Selection)")
    dist1, dist2 = st.columns(2)

    with dist1:
        # Age Distribution
        age_dist = alt.Chart(filtered_df).mark_bar(color='#4c78a8').encode(
            x=alt.X("age:Q", bin=alt.Bin(maxbins=20), title="Patient Age at Diagnosis"),
            y=alt.Y("count():Q", title="Patient Frequency"),
            tooltip=["age", "count()"]
        ).properties(height=220, title="Age Distribution")
        st.altair_chart(age_dist, width="stretch")

    with dist2:
        # Stage Prevalence
        stage_counts = filtered_df["stage"].value_counts().reset_index()
        stage_counts.columns = ["Stage", "Count"]
        stage_dist = alt.Chart(stage_counts).mark_bar(color='#72b7b2').encode(
            x=alt.X("Count:Q", title="Number of Patients"),
            y=alt.Y("Stage:N", sort='-x', title="AJCC Stage"),
            tooltip=["Stage", "Count"]
        ).properties(height=220, title="Stage Prevalence")
        st.altair_chart(stage_dist, width="stretch")

    st.divider()

    # --- 3. Treatment Analysis Row ---
    c1, c2, c3 = st.columns([1, 2.5, 2], gap="large")

    with c1:
        st.write("**Specific Therapy Choice:**")
        therapy_choice = st.radio(
            "Select treatment modality", 
            ["Surgery", "Chemotherapy", "Radiation Therapy", "Hormone Therapy"],
            label_visibility="collapsed"
        )
        
        mapping = {
            "Surgery": "Surgery, NOS",
            "Chemotherapy": "Chemotherapy|Pharmaceutical Therapy, NOS",
            "Radiation Therapy": "Radiation Therapy, NOS|Radiation, External Beam",
            "Hormone Therapy": "Hormone Therapy"
        }
        
        cohort_data = filtered_df[filtered_df["treatment"].str.contains(mapping[therapy_choice], case=False, na=False)]
        st.metric("Patient Cohort Count", len(cohort_data))

    with c2:
        if len(cohort_data) >= 3:
            # Passing therapy_choice to handle the KM label fallback
            st.pyplot(plot_km_brca(cohort_data, therapy_choice))
        else:
            st.warning("Sample size is too small to calculate survival probability.")

    with c3:
        if not cohort_data.empty:
            st.write("**Clinical Category Profile**")
            profile_field = "agent" if cohort_data["agent"].nunique() > 1 else "treatment"
            counts = cohort_data[profile_field].value_counts().nlargest(8).reset_index()
            counts.columns = ['Sub-Category', 'Patient Count']
            
            donut_viz = alt.Chart(counts).mark_arc(innerRadius=50).encode(
                theta="Patient Count", color="Sub-Category", tooltip=['Sub-Category', 'Patient Count']
            ).properties(height=300)
            st.altair_chart(donut_viz, width="stretch")

if __name__ == "__main__":
    main()