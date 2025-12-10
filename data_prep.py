import numpy as np
import pandas as pd
import streamlit as st

# --- Data Cleaning Constants & Utilities ---
BAD_TOKENS = {"'--", "--", "NA", "Na", "N/A", "Not Reported", "Unknown", "", "nan", "NaN"}


def clean_string_series(s: pd.Series) -> pd.Series:
    """Convert bad tokens to NaN in string series."""
    return s.astype(str).str.strip().replace(list(BAD_TOKENS), np.nan)


def clean_numeric_series(s: pd.Series) -> pd.Series:
    """Convert bad tokens to NaN and coerce to numeric."""
    return pd.to_numeric(s.astype(str).replace(list(BAD_TOKENS), np.nan), errors="coerce")


@st.cache_data
def load_and_preprocess_brca(path: str) -> pd.DataFrame:
    """
    Load clinical.tsv and preprocess for BRCA analysis.
    
    Returns:
        DataFrame with derived survival variables and cleaned clinical fields.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    
    # Isolate Breast cancer cohort
    df["tumor_type"] = clean_string_series(df.get("diagnoses.tissue_or_organ_of_origin", pd.Series("Other")))
    brca_df = df[df["tumor_type"].str.contains("Breast", case=False, na=False)].copy()
    
    # Survival & Event status
    if "demographic.vital_status" in brca_df.columns:
        brca_df["event"] = brca_df["demographic.vital_status"].map({"Dead": 1, "Alive": 0}).fillna(0).astype(int)
    else:
        brca_df["event"] = 0
    
    # Time to event: use death if available, otherwise follow-up
    death = clean_numeric_series(brca_df.get("demographic.days_to_death", pd.Series(np.nan)))
    follow = clean_numeric_series(brca_df.get("diagnoses.days_to_last_follow_up", pd.Series(np.nan)))
    brca_df["time"] = np.where(death.notna(), death, follow)
    
    # Clinical variables
    brca_df["age"] = clean_numeric_series(brca_df.get("demographic.age_at_index", pd.Series(np.nan)))
    brca_df["stage"] = clean_string_series(brca_df.get("diagnoses.ajcc_pathologic_stage", pd.Series("Unknown")))
    brca_df["treatment"] = clean_string_series(brca_df.get("treatments.treatment_type", pd.Series("Other")))
    brca_df["agent"] = clean_string_series(brca_df.get("treatments.therapeutic_agents", pd.Series("Not Specified")))
    
    # Demographics for Page B (sites & demographics)
    brca_df["race"] = clean_string_series(brca_df.get("demographic.race", pd.Series("Unknown")))
    brca_df["ethnicity"] = clean_string_series(brca_df.get("demographic.ethnicity", pd.Series("Unknown")))
    brca_df["gender"] = clean_string_series(brca_df.get("demographic.gender", pd.Series("Unknown")))
    brca_df["site"] = clean_string_series(brca_df.get("diagnoses.tissue_or_organ_of_origin", pd.Series("Unknown")))

    # Country (prefer residence, fallback to birth)
    country_res = clean_string_series(brca_df.get("demographic.country_of_residence_at_enrollment", pd.Series(np.nan)))
    country_birth = clean_string_series(brca_df.get("demographic.country_of_birth", pd.Series(np.nan)))
    brca_df["country"] = country_res.fillna(country_birth)
    
    return brca_df.dropna(subset=["time"])
