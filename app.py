import math
import re
import xml.etree.ElementTree as ET
from base64 import b64encode
from collections import Counter, defaultdict

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lifelines import KaplanMeierFitter
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

st.set_page_config(layout="wide", page_title="TCGA BRCA Dashboard")

BAD_TOKENS = {
    "'--",
    "--",
    "NA",
    "Na",
    "N/A",
    "Not Reported",
    "Unknown",
    "",
    "nan",
    "NaN",
}


def clean_string_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace(list(BAD_TOKENS), np.nan)


def clean_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).replace(list(BAD_TOKENS), np.nan), errors="coerce"
    )


@st.cache_data
def load_and_preprocess_brca(path: str, drop_missing_time: bool = True):
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    df["tumor_type"] = clean_string_series(
        df.get("diagnoses.tissue_or_organ_of_origin", pd.Series("Other"))
    )
    brca_df = df[df["tumor_type"].str.contains("Breast", case=False, na=False)].copy()
    if "demographic.vital_status" in brca_df.columns:
        brca_df["event"] = (
            brca_df["demographic.vital_status"]
            .map({"Dead": 1, "Alive": 0})
            .fillna(0)
            .astype(int)
        )
    death = clean_numeric_series(
        brca_df.get("demographic.days_to_death", pd.Series(np.nan))
    )
    follow = clean_numeric_series(
        brca_df.get("diagnoses.days_to_last_follow_up", pd.Series(np.nan))
    )
    brca_df["time"] = np.where(death.notna(), death, follow)
    brca_df["age"] = clean_numeric_series(
        brca_df.get("demographic.age_at_index", pd.Series(np.nan))
    )
    brca_df["stage"] = clean_string_series(
        brca_df.get("diagnoses.ajcc_pathologic_stage", pd.Series("Unknown"))
    )
    brca_df["treatment"] = clean_string_series(
        brca_df.get("treatments.treatment_type", pd.Series("Other"))
    )
    brca_df["agent"] = clean_string_series(
        brca_df.get("treatments.therapeutic_agents", pd.Series("Not Specified"))
    )
    brca_df["race"] = clean_string_series(
        brca_df.get("demographic.race", pd.Series("Unknown"))
    )
    brca_df["ethnicity"] = clean_string_series(
        brca_df.get("demographic.ethnicity", pd.Series("Unknown"))
    )
    brca_df["site"] = clean_string_series(
        brca_df.get("cases.primary_site", pd.Series("Unknown"))
    )
    brca_df["country"] = clean_string_series(
        brca_df.get(
            "demographic.country_of_residence_at_enrollment", pd.Series("Unknown")
        )
    )
    
    if drop_missing_time:
        brca_df = brca_df.dropna(subset=["time"])
    return brca_df


@st.cache_data
def load_and_process_data():
    try:
        clin = pd.read_csv("clinical.tsv", sep="\t", low_memory=False)

        def _find_patient_column(df):
            for candidate in [
                "cases.submitter_id",
                "submitter_id",
                "case_id",
                "case_submitter_id",
            ]:
                if candidate in df.columns:
                    return candidate
            for col in df.columns:
                try:
                    ser = df[col].astype(str)
                except Exception:
                    continue
                frac = ser.str.match(r"^TCGA-[A-Za-z0-9]+-[A-Za-z0-9]+$").sum() / max(
                    1, len(ser)
                )
                if frac > 0.05:
                    return col
            return None

        patient_col = _find_patient_column(clin)
        if patient_col is None:
            st.error("❌ Could not find a TCGA patient ID column in 'clinical.tsv'.")
            return None, None
        clin["patientID"] = (
            clin[patient_col]
            .astype(str)
            .str.extract(r"^(TCGA-[A-Za-z0-9]+-[A-Za-z0-9]+)")
        )
        clin = clin.drop_duplicates(subset="patientID")
        if "diagnoses.ajcc_pathologic_stage" in clin.columns:
            clin["Subtype_Proxy"] = clin["diagnoses.ajcc_pathologic_stage"].fillna(
                "Unknown"
            )
        else:
            clin["Subtype_Proxy"] = "Unknown"
    except FileNotFoundError:
        st.error("❌ 'clinical.tsv' not found.")
        return None, None
    try:
        expr = pd.read_csv("expression.txt", sep="\t", index_col=0, nrows=2000)
        st.success(f"✅ Loaded {expr.shape[0]} genes and {expr.shape[1]} samples.")
    except FileNotFoundError:
        st.warning("⚠️ 'expression.txt' not found. Using Mock Data.")
        return generate_mock_data()

    def _sample_to_patient(sample_id: str) -> str:
        parts = str(sample_id).split("-")
        if len(parts) >= 3:
            return "-".join(parts[:3])
        return str(sample_id)

    expr_patient_map = {c: _sample_to_patient(c) for c in expr.columns}
    clin_patients = set(clin["patientID"].dropna().astype(str).values)
    common_samples = [c for c, pid in expr_patient_map.items() if pid in clin_patients]
    expr = expr[common_samples]
    clin = clin[clin["patientID"].isin({expr_patient_map[c] for c in common_samples})]
    variances = expr.var(axis=1)
    top_genes = variances.sort_values(ascending=False).head(50).index
    expr_filtered = expr.loc[top_genes]
    expr_t = expr_filtered.T.reset_index().rename(columns={"index": "sampleID"})
    expr_t["patientID"] = expr_t["sampleID"].astype(str).map(_sample_to_patient)
    df_merged = pd.merge(
        expr_t, clin[["patientID", "Subtype_Proxy"]], on="patientID", how="left"
    )
    df_long = df_merged.melt(
        id_vars=["sampleID", "patientID", "Subtype_Proxy"],
        var_name="Gene",
        value_name="Expression",
    )
    return df_merged, df_long


def generate_mock_data():
    np.random.seed(42)
    df_samples = pd.DataFrame(
        {
            "sampleID": [f"S{i}" for i in range(100)],
            "Subtype_Proxy": np.random.choice(["Stage I", "Stage II"], 100),
        }
    )
    df_long = pd.DataFrame(
        {
            "sampleID": ["S0"] * 10,
            "Gene": ["G1"] * 10,
            "Expression": np.random.randn(10),
            "Subtype_Proxy": ["Stage I"] * 10,
        }
    )
    return df_samples, df_long


@st.cache_data
def load_clinical_for_anatomy():
    clinical = pd.read_csv("data/clinical.tsv", sep="\t", low_memory=False)
    clinical.replace("'--", pd.NA, inplace=True)
    return clinical


def apply_color_to_element(elem, hex_color):
    shape_tags = ("path", "rect", "circle", "ellipse", "polygon", "polyline")
    for node in elem.iter():
        tag = node.tag.split("}")[-1]
        if tag in shape_tags:
            style = node.get("style", "")
            style = re.sub(r"fill:[^;]+;?", "", style)
            style = style.strip().rstrip(";")
            if style:
                style += ";"
            style += f"fill:{hex_color}"
            node.set("style", style)
            node.set("fill", hex_color)


def generate_colored_svg_data_uri(
    SVG_TEMPLATE_BYTES, prevalence_dict, active_organ=None, co_occurrence_dict=None
):
    root = ET.fromstring(SVG_TEMPLATE_BYTES)
    quadrant_mapping = {
        "left_upper_inner": "quadrant_path_1",
        "left_lower_inner": "quadrant_path_2",
        "left_lower_outer": "quadrant_path_3",
        "left_upper_outer": "quadrant_path_4",
        "right_upper_inner": "quadrant_path_5",
        "right_upper_outer": "quadrant_path_6",
        "right_lower_outer": "quadrant_path_7",
        "right_lower_inner": "quadrant_path_8",
    }
    quadrant_d_patterns = {
        "M433.9,312.7": "quadrant_path_1",
        "M374.6,371.1": "quadrant_path_2",
        "M341.8,370.2": "quadrant_path_3",
        "M285.5,313.6": "quadrant_path_4",
        "M465.7,312.7": "quadrant_path_5",
        "M612.5,313.2": "quadrant_path_6",
        "M558.2,370.2": "quadrant_path_7",
        "M524.1,371.1": "quadrant_path_8",
    }
    organ_to_group_id = {
        "abdomen": "group_abdomen",
        "bone": "group_bone",
        "skin": "group_skin",
        "lymph_node": "group_lymph_node",
        "ovary": "group_ovary",
        "adrenal_gland": "group_adrenal_gland",
    }
    for child in root:
        tag = child.tag.split("}")[-1]
        if tag == "path":
            d_attr = child.get("d", "")
            if "M114.5,512.9" in d_attr:
                child.set("id", "liver_path")
            elif "M86.2,312.8" in d_attr:
                child.set("id", "lung_path")
            else:
                for pattern, quad_id in quadrant_d_patterns.items():
                    if d_attr.startswith(pattern):
                        child.set("id", quad_id)
                        break
    organ_to_group_id["liver"] = "liver_path"
    organ_to_group_id["lung"] = "lung_path"
    organ_to_group_id["breast_quadrants_background"] = "group_breast_quadrants"
    name_to_ids = defaultdict(list)
    for organ_name, group_id in organ_to_group_id.items():
        name_to_ids[organ_name].append(group_id)
    for qname, path_id in quadrant_mapping.items():
        name_to_ids[qname].append(path_id)
    for name, val in prevalence_dict.items():
        hex_color = None
        if active_organ is not None:
            if name == active_organ:
                hex_color = prevalence_to_hex(val, name, is_active=True)
            elif co_occurrence_dict and active_organ in co_occurrence_dict:
                co_val = co_occurrence_dict[active_organ].get(name)
                if co_val is not None and co_val > 0:
                    hex_color = prevalence_to_hex(val, name, co_occurrence_value=co_val)
        else:
            hex_color = prevalence_to_hex(val, name)
        if hex_color:
            for eid in name_to_ids.get(name, []):
                elem = root.find(f".//*[@id='{eid}']")
                if elem is not None:
                    apply_color_to_element(elem, hex_color)
    svg_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    b64 = b64encode(svg_bytes).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


organ_positions = {
    "left_upper_inner": (405, 270),
    "left_lower_inner": (360, 345),
    "left_lower_outer": (315, 355),
    "left_upper_outer": (312, 270),
    "right_upper_inner": (495, 270),
    "right_upper_outer": (585, 270),
    "right_lower_outer": (583, 345),
    "right_lower_inner": (540, 345),
    "abdomen": (450, 490),
    "lung": (100, 350),
    "liver": (100, 540),
    "adrenal_gland": (83, 755),
    "bone": (770, 370),
    "skin": (770, 540),
    "lymph_node": (775, 745),
    "ovary": (445, 720),
}
organs = list(organ_positions.keys())
SITE_TO_ORGAN = {
    "Breast, Left Upper Outer": "left_upper_outer",
    "Breast, Left Upper Inner": "left_upper_inner",
    "Breast, Left Lower Outer": "left_lower_outer",
    "Breast, Left Lower Inner": "left_lower_inner",
    "Breast, Right Upper Outer": "right_upper_outer",
    "Breast, Right Upper Inner": "right_upper_inner",
    "Breast, Right Lower Outer": "right_lower_outer",
    "Breast, Right Lower Inner": "right_lower_inner",
    "Lung, NOS": "lung",
    "Liver": "liver",
    "Ovary, NOS": "ovary",
    "Lymph node, NOS": "lymph_node",
    "Adrenal gland, NOS": "adrenal_gland",
    "Skin": "skin",
    "Abdomen": "abdomen",
    "Bone, NOS": "bone",
    "Breast, NOS": None,
}
BREAST_QUADRANTS = {
    "left_upper_outer",
    "left_upper_inner",
    "left_lower_outer",
    "left_lower_inner",
    "right_upper_outer",
    "right_upper_inner",
    "right_lower_outer",
    "right_lower_inner",
}


def map_site_to_organ(site):
    return SITE_TO_ORGAN.get(site)


def load_prevalence_and_cooccurrence():
    clinical = load_clinical_for_anatomy()
    case_organs = []
    for _, row in clinical.iterrows():
        sites_str = row.get("diagnoses.sites_of_involvement")
        if pd.notna(sites_str):
            sites = [s.strip() for s in sites_str.split("|")]
            mapped = [map_site_to_organ(s) for s in sites]
            mapped = [o for o in mapped if o is not None]
            if mapped:
                case_organs.append((row.get("cases.case_id"), mapped))
    all_organs_list = [organ for _, organs_list in case_organs for organ in organs_list]
    organ_counts = pd.Series(all_organs_list).value_counts()
    total_cases = len(case_organs)
    prevalence = {
        organ: organ_counts[organ] / total_cases for organ in organ_counts.index
    }
    counts = {organ: organ_counts[organ] for organ in organ_counts.index}
    all_organs_set = set(prevalence.keys())
    co_occurrence = {}
    for organ_a in all_organs_set:
        co_occurrence[organ_a] = {}
        cases_with_a = sum(
            1 for _, organs_list in case_organs if organ_a in organs_list
        )
        for organ_b in all_organs_set:
            if organ_a != organ_b:
                cases_with_both = sum(
                    1
                    for _, organs_list in case_organs
                    if organ_a in organs_list and organ_b in organs_list
                )
                co_occurrence[organ_a][organ_b] = (
                    cases_with_both / cases_with_a if cases_with_a > 0 else 0.0
                )
    return clinical, prevalence, co_occurrence, counts


CLINICAL_ANATOMY, PREVALENCE, CO_OCCURRENCE, COUNTS = load_prevalence_and_cooccurrence()
with open("data/anatomy.svg", "rb") as f:
    SVG_TEMPLATE_BYTES = f.read()


def prevalence_to_hex(p, organ_name, is_active=False, co_occurrence_value=None):
    p = max(0.0, min(1.0, p))
    if is_active:
        return "#0066cc"
    if co_occurrence_value is not None:
        co_val = max(0.0, min(1.0, co_occurrence_value))
        co_val_scaled = math.log10(1 + co_val * 99) / 2 if co_val > 0 else 0
        if organ_name in BREAST_QUADRANTS:
            r = int(255 + co_val_scaled * (150 - 255))
            g = int(230 + co_val_scaled * (0 - 230))
            b = int(230 + co_val_scaled * (0 - 230))
        else:
            r = int(230 + co_val_scaled * (0 - 230))
            g = int(255 + co_val_scaled * (150 - 255))
            b = int(230 + co_val_scaled * (0 - 230))
        return f"#{r:02x}{g:02x}{b:02x}"
    p_scaled = math.log10(1 + p * 99) / 2 if p > 0 else 0
    if organ_name in BREAST_QUADRANTS:
        r = int(255 + p_scaled * (150 - 255))
        g = int(230 + p_scaled * (0 - 230))
        b = int(230 + p_scaled * (0 - 230))
    else:
        r = int(230 + p_scaled * (0 - 230))
        g = int(255 + p_scaled * (150 - 255))
        b = int(230 + p_scaled * (0 - 230))
    return f"#{r:02x}{g:02x}{b:02x}"


def make_figure(active_organ=None):
    bg_src = generate_colored_svg_data_uri(
        SVG_TEMPLATE_BYTES,
        PREVALENCE,
        active_organ=active_organ,
        co_occurrence_dict=CO_OCCURRENCE,
    )
    scale_x = 500 / 859
    scale_y = 500 / 840.6
    xs, ys, texts, customdata = [], [], [], []
    for organ in organs:
        x, y = organ_positions[organ]
        xs.append(x * scale_x)
        ys.append(y * scale_y)
        texts.append(organ)
        organ_display = organ.replace("_", " ").title()
        count = COUNTS.get(organ, 0)
        hover_text = f"<b>{organ_display}</b><br>Count: {count}<br>"
        if active_organ and organ == active_organ:
            hover_text += "<b>Selected organ</b>"
        elif active_organ and CO_OCCURRENCE.get(active_organ, {}).get(organ):
            co_val = CO_OCCURRENCE[active_organ][organ]
            hover_text += f"Co-occurrence with {active_organ.replace('_', ' ').title()}: {co_val:.2%}"
        customdata.append(hover_text)
    scatter = go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        text=texts,
        textposition="top center",
        marker=dict(size=60, color="rgba(0,0,0,0)", line=dict(width=0)),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=customdata,
        name="Organs",
    )
    fig = go.Figure(data=[scatter])
    fig.update_xaxes(range=[0, 500], visible=False)
    fig.update_yaxes(range=[500, 0], visible=False)
    fig.update_layout(
        width=500,
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False,
        plot_bgcolor="white",
    )
    fig.add_layout_image(
        dict(
            source=bg_src,
            xref="paper",
            yref="paper",
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            xanchor="left",
            yanchor="top",
            sizing="stretch",
            opacity=1.0,
            layer="below",
        )
    )
    shapes = []
    annotations = []
    if active_organ is not None and active_organ in CO_OCCURRENCE:
        x0, y0 = organ_positions[active_organ]
        x0_scaled = x0 * scale_x
        y0_scaled = y0 * scale_y
        targets = CO_OCCURRENCE[active_organ]
        vals = [v for v in targets.values() if v is not None and v > 0]
        if vals:
            vmin, vmax = min(vals), max(vals)
            if vmax == vmin:
                vmax = vmin + 1e-6
            for target, val in targets.items():
                if val <= 0 or target not in organ_positions:
                    continue
                x1, y1 = organ_positions[target]
                x1_scaled = x1 * scale_x
                y1_scaled = y1 * scale_y
                t = (val - vmin) / (vmax - vmin + 1e-9)
                width = 2 + 8 * t
                annotations.append(
                    dict(
                        x=x1_scaled,
                        y=y1_scaled,
                        ax=x0_scaled,
                        ay=y0_scaled,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=0.5,
                        arrowwidth=width,
                        arrowcolor=f"rgba(50, 100, 200, {0.5 + 0.4 * t})",
                        text="",
                    )
                )
    fig.update_layout(shapes=shapes, annotations=annotations)
    return fig


def get_stage_data(selected_organ=None):
    if selected_organ is None:
        return CLINICAL_ANATOMY["diagnoses.ajcc_pathologic_stage"].value_counts()
    filtered_cases = []
    for _, row in CLINICAL_ANATOMY.iterrows():
        sites_str = row.get("diagnoses.sites_of_involvement")
        if pd.notna(sites_str):
            sites = [s.strip() for s in sites_str.split("|")]
            organs_for_case = [map_site_to_organ(s) for s in sites]
            if selected_organ in organs_for_case:
                filtered_cases.append(
                    {
                        "diagnoses.ajcc_pathologic_stage": row.get(
                            "diagnoses.ajcc_pathologic_stage"
                        )
                    }
                )
    if not filtered_cases:
        return pd.Series(dtype=int)
    filtered_df = pd.DataFrame(filtered_cases)
    return filtered_df["diagnoses.ajcc_pathologic_stage"].value_counts()


def make_stage_chart(selected_organ=None):
    stage_data = get_stage_data(selected_organ)
    stage_order = [
        "Stage 0",
        "Stage 0is",
        "Stage I",
        "Stage IA",
        "Stage IB",
        "Stage II",
        "Stage IIA",
        "Stage IIB",
        "Stage III",
        "Stage IIIA",
        "Stage IIIB",
        "Stage IIIC",
        "Stage IV",
        "Stage X",
    ]
    ordered_stages = [s for s in stage_order if s in stage_data.index]
    ordered_counts = [stage_data[s] for s in ordered_stages]
    total_cases = sum(ordered_counts)
    hover_texts = []
    for stage, count in zip(ordered_stages, ordered_counts):
        percentage = (count / total_cases * 100) if total_cases > 0 else 0
        hover_text = (
            f"<b>{stage}</b><br>Cases: {count}<br>Percentage: {percentage:.1f}%"
        )
        if selected_organ:
            hover_text += f"<br>For: {selected_organ.replace('_', ' ').title()}"
        hover_texts.append(hover_text)
    fig = go.Figure(
        data=[
            go.Bar(
                x=ordered_stages,
                y=ordered_counts,
                marker=dict(color="steelblue"),
                text=ordered_counts,
                textposition="outside",
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts,
            )
        ]
    )
    title_text = "Tumor Stage Distribution"
    if selected_organ:
        organ_display = selected_organ.replace("_", " ").title()
        title_text = f"Tumor Stage Distribution: {organ_display}"
    fig.update_layout(
        height=550,
        margin=dict(l=40, r=40, t=60, b=100),
        title=title_text,
        xaxis_title="Stage",
        yaxis_title="Number of Cases",
        showlegend=False,
    )
    return fig


def make_upset_plot(selected_organ=None):
    combinations = []
    for _, row in CLINICAL_ANATOMY.iterrows():
        sites_str = row.get("diagnoses.sites_of_involvement")
        if pd.notna(sites_str):
            sites = [s.strip() for s in sites_str.split("|")]
            mapped = [map_site_to_organ(s) for s in sites]
            mapped = [o for o in mapped if o is not None]
            if mapped:
                combinations.append(frozenset(mapped))
    combo_counts = Counter(combinations)
    top_combos = sorted(combo_counts.items(), key=lambda x: -x[1])[:20]
    all_organs_in_combos = set()
    for combo, _ in top_combos:
        all_organs_in_combos.update(combo)
    organs_list = sorted(all_organs_in_combos)
    n_combos = len(top_combos)
    n_organs = len(organs_list)
    highlight_indices = []
    if selected_organ:
        for i, (combo, _) in enumerate(top_combos):
            if selected_organ in combo:
                highlight_indices.append(i)
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.05,
        subplot_titles=("", ""),
        specs=[[{"type": "bar"}], [{"type": "scatter"}]],
    )
    counts = [count for _, count in top_combos]
    bar_colors = [
        "#ff6b6b" if i in highlight_indices else "steelblue" for i in range(n_combos)
    ]
    combo_hover_texts = []
    for combo, count in top_combos:
        organs_display = "<br>".join(
            [o.replace("_", " ").title() for o in sorted(combo)]
        )
        combo_hover_texts.append(
            f"<b>Combination ({len(combo)} organs):</b><br>{organs_display}<br><b>Count:</b> {count}"
        )
    fig.add_trace(
        go.Bar(
            x=list(range(n_combos)),
            y=counts,
            marker=dict(color=bar_colors),
            showlegend=False,
            text=counts,
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=combo_hover_texts,
            name="Count",
        ),
        row=1,
        col=1,
    )
    for j, (combo, count) in enumerate(top_combos):
        organs_in_combo = [organs_list.index(o) for o in combo if o in organs_list]
        if len(organs_in_combo) > 1:
            organs_in_combo.sort()
            for i in range(len(organs_in_combo) - 1):
                is_highlighted = j in highlight_indices
                fig.add_trace(
                    go.Scatter(
                        x=[j, j],
                        y=[organs_in_combo[i], organs_in_combo[i + 1]],
                        mode="lines",
                        line=dict(
                            color="#ff6b6b" if is_highlighted else "#4a4a4a",
                            width=3 if is_highlighted else 2,
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=2,
                    col=1,
                )
    for i, organ in enumerate(organs_list):
        x_coords = []
        y_coords = []
        hover_texts = []
        for j, (combo, count) in enumerate(top_combos):
            if organ in combo:
                x_coords.append(j)
                y_coords.append(i)
                other_organs = [
                    o.replace("_", " ").title() for o in sorted(combo) if o != organ
                ]
                hover_text = (
                    f"<b>{organ.replace('_', ' ').title()}</b><br>Co-occurs with:<br>"
                    + "<br>".join(other_organs)
                    + f"<br><b>Total count:</b> {count}"
                )
                hover_texts.append(hover_text)
        is_selected = organ == selected_organ
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker=dict(
                    size=12,
                    color="#ff6b6b" if is_selected else "#4a4a4a",
                    symbol="circle",
                    line=dict(width=1, color="white"),
                ),
                name=organ,
                showlegend=False,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts,
            ),
            row=2,
            col=1,
        )
    fig.update_xaxes(
        title="", range=[-0.5, n_combos - 0.5], showticklabels=False, row=1, col=1
    )
    fig.update_yaxes(title="Count", row=1, col=1)
    fig.update_xaxes(
        title="", range=[-0.5, n_combos - 0.5], showticklabels=False, row=2, col=1
    )
    fig.update_yaxes(
        title="Organs",
        ticktext=[o.replace("_", " ").title() for o in organs_list],
        tickvals=list(range(n_organs)),
        range=[-0.5, n_organs - 0.5],
        row=2,
        col=1,
    )
    fig.update_layout(
        height=500,
        margin=dict(l=150, r=40, t=60, b=60),
        title="Upset Plot (Top 20)"
        + (
            f" - Highlighting: {selected_organ.replace('_', ' ').title()}"
            if selected_organ
            else ""
        ),
        hovermode="closest",
        showlegend=False,
    )
    return fig


def page_expression():
    st.header("Gene Expression Explorer")
    df_samples, df_long = load_and_process_data()
    if df_samples is None:
        return
    all_genes = [g for g in df_long["Gene"].unique()]
    gene_options = sorted(all_genes) if all_genes else ["N/A"]
    subtype_options = sorted(df_samples["Subtype_Proxy"].dropna().unique())

    control_cols = st.columns(2)
    with control_cols[0]:
        cluster_method = st.selectbox(
            "Clustering linkage method",
            ["ward", "average", "complete", "single"],
            index=0,
            key="expr_cluster",
        )
        distance_metric = st.selectbox(
            "Distance metric",
            ["euclidean", "cityblock", "cosine"],
            index=0,
            key="expr_dist",
        )
        heatmap_filter_mode = st.radio(
            "Heatmap selection mode",
            ["Brush from scatter", "Click bar -> highlight"],
            index=0,
            key="expr_heatmap_mode",
        )
        scatter_mode = st.radio(
            "Scatter mode",
            ["UMAP", "Manual gene axes"],
            index=0,
            key="expr_scatter_mode",
        )
        g1_manual = None
        g2_manual = None
        if scatter_mode == "Manual gene axes":
            g1_manual = st.selectbox(
                "Scatter X gene", options=gene_options, index=0, key="expr_g1"
            )
            g2_manual = st.selectbox(
                "Scatter Y gene",
                options=gene_options,
                index=1 if len(gene_options) > 1 else 0,
                key="expr_g2",
            )

    with control_cols[1]:
        select_all_subtypes = st.checkbox(
            "Select all subtypes", value=True, key="expr_all_subtypes"
        )
        selected_subtypes = st.multiselect(
            "Filter by subtype",
            subtype_options,
            default=subtype_options,
            key="expr_subtypes",
        )
        if select_all_subtypes:
            selected_subtypes = subtype_options
        select_all_genes = st.checkbox(
            "Select all genes (shown)", value=False, key="expr_all_genes"
        )
        gene_filter = st.multiselect(
            "Limit heatmap to genes",
            options=gene_options,
            default=gene_options[: min(20, len(gene_options))],
            help="Choose subset to speed rendering; gene search always adds a match",
            key="expr_gene_filter",
        )
        if select_all_genes:
            gene_filter = gene_options

    plots_tab, gene_tab = st.tabs(["Plots", "Gene Card"])
    gene_search = ""
    with gene_tab:
        gene_search = st.text_input(
            "Find Gene (exact match)", "", key="expr_gene_search"
        )
    if selected_subtypes:
        df_samples = df_samples[df_samples["Subtype_Proxy"].isin(selected_subtypes)]
        df_long = df_long[df_long["Subtype_Proxy"].isin(selected_subtypes)]
    heatmap_genes = {g for g in gene_filter if g != "N/A"}
    if gene_search and gene_search in all_genes:
        heatmap_genes.add(gene_search)
    if heatmap_genes:
        df_long = df_long[df_long["Gene"].isin(heatmap_genes)]
    umap_ready = False
    umap_cols = [
        c
        for c in df_samples.columns
        if c not in ["sampleID", "patientID", "Subtype_Proxy"]
    ]
    if len(umap_cols) >= 2:
        try:
            import umap

            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            embedding = reducer.fit_transform(df_samples[umap_cols].fillna(0))
            df_samples["UMAP1"], df_samples["UMAP2"] = embedding[:, 0], embedding[:, 1]
            umap_ready = True
        except Exception:
            umap_ready = False
    matrix = df_long.pivot(index="Gene", columns="sampleID", values="Expression")
    if matrix.empty:
        st.error(
            "❌ No overlapping expression/clinical samples available for visualization."
        )
        return
    gene_order = matrix.index.tolist()
    if gene_search in gene_order:
        gene_order = [gene_search] + [g for g in gene_order if g != gene_search]
    sample_order = matrix.columns.tolist()
    if matrix.shape[0] >= 2:
        row_link = linkage(pdist(matrix, metric=distance_metric), method=cluster_method)
        gene_order = matrix.index[leaves_list(row_link)].tolist()
    else:
        st.warning(
            "⚠️ Need at least two genes to compute clustering. Showing raw order."
        )
    if matrix.shape[1] >= 2:
        col_link = linkage(
            pdist(matrix.T, metric=distance_metric), method=cluster_method
        )
        sample_order = matrix.columns[leaves_list(col_link)].tolist()
    else:
        st.warning(
            "⚠️ Need at least two samples to compute clustering. Showing raw order."
        )
    brush = None
    scatter = None
    bar_selection = alt.selection_point(
        fields=["Subtype_Proxy"], toggle="true", empty="all"
    )
    gene_bar_selection = alt.selection_point(
        fields=["Gene"], toggle="true", empty="all"
    )
    scatter = None
    brush = None
    if len(gene_order) >= 2:
        if scatter_mode == "Manual gene axes":
            g1 = g1_manual or gene_order[0]
            g2 = g2_manual or (gene_order[1] if len(gene_order) > 1 else gene_order[0])
        else:
            g1, g2 = gene_order[0], gene_order[1]
        for fallback_gene in gene_order:
            if g1 not in df_samples.columns:
                g1 = fallback_gene
            if g2 not in df_samples.columns:
                g2 = fallback_gene
        scatter_cols = ["sampleID", "Subtype_Proxy", g1, g2]
        if scatter_mode == "UMAP" and umap_ready and "UMAP1" in df_samples.columns:
            scatter_cols += ["UMAP1", "UMAP2"]
        scatter_df = df_samples[scatter_cols].copy()
        df_long = df_long.merge(
            scatter_df[["sampleID", g1, g2]], how="left", on="sampleID"
        )
        brush = alt.selection_interval(encodings=["x", "y"])
        scatter_base = alt.Chart(scatter_df)
        if scatter_mode == "UMAP" and umap_ready and "UMAP1" in df_samples.columns:
            scatter = (
                scatter_base.mark_circle(size=70, opacity=0.8)
                .encode(
                    x=alt.X("UMAP1", title="UMAP1"),
                    y=alt.Y("UMAP2", title="UMAP2"),
                    color=alt.Color("Subtype_Proxy", legend=None),
                    tooltip=["sampleID", "Subtype_Proxy", g1, g2],
                )
                .add_params(brush)
                .properties(title="Sample Similarity (UMAP)", width=400, height=300)
            )
        else:
            scatter = (
                scatter_base.mark_circle(size=70, opacity=0.8)
                .encode(
                    x=alt.X(g1, title=f"{g1} Expression"),
                    y=alt.Y(g2, title=f"{g2} Expression"),
                    color=alt.Color("Subtype_Proxy", legend=None),
                    tooltip=["sampleID", "Subtype_Proxy", g1, g2],
                )
                .add_params(brush)
                .properties(
                    title="Sample Similarity (Gene-Gene)", width=400, height=300
                )
            )
        if heatmap_filter_mode == "Click bar -> highlight":
            scatter = scatter.transform_filter(bar_selection)
    else:
        st.info("ℹ️ At least two genes are required to draw the scatter plot.")

    subtype_bar = (
        alt.Chart(df_samples)
        .mark_bar()
        .encode(
            x=alt.X("Subtype_Proxy", sort="-y", title="Subtype"),
            y=alt.Y("count()", title="Count"),
            color=alt.Color("Subtype_Proxy", legend=None),
            tooltip=["Subtype_Proxy", alt.Tooltip("count()", title="Samples")],
        )
        .add_params(bar_selection)
        .properties(title="Subtype counts", width=400, height=150)
    )

    gene_var = (
        df_long.groupby("Gene")["Expression"]
        .var()
        .sort_values(ascending=False)
        .reset_index()
    )

    top_gene_bar = (
        alt.Chart(gene_var)
        .mark_bar()
        .encode(
            x=alt.X("Gene", sort="-y"),
            y=alt.Y("Expression", title="Variance"),
            color=alt.value("#4c78a8"),
            tooltip=["Gene", alt.Tooltip("Expression", title="Variance")],
        )
        .add_params(gene_bar_selection)
        .properties(title="Top genes by variance", width=850, height=200)
    )

    heatmap = (
        alt.Chart(df_long)
        .mark_rect()
        .encode(
            x=alt.X("sampleID", sort=sample_order, axis=None),
            y=alt.Y("Gene", sort=gene_order),
            color=alt.Color("Expression", scale=alt.Scale(scheme="redblue")),
            tooltip=["sampleID", "Gene", "Expression", "Subtype_Proxy"],
        )
        .properties(title="Top 50 Variable Genes", width=400)
    )

    if heatmap_filter_mode == "Brush from scatter" and brush is not None:
        heatmap = heatmap.transform_filter(brush)
    if heatmap_filter_mode == "Click bar -> highlight":
        heatmap = heatmap.transform_filter(bar_selection)
    heatmap = heatmap.transform_filter(gene_bar_selection)

    if scatter is not None:
        left_col = (scatter & subtype_bar).resolve_scale(color="independent")
    else:
        left_col = subtype_bar

    top_row = alt.hconcat(left_col, heatmap).resolve_scale(color="independent")
    final_chart = alt.vconcat(top_row, top_gene_bar).resolve_scale(color="independent")

    with plots_tab:
        st.info("Loading the plots may take a moment...")
        st.altair_chart(final_chart, use_container_width=False)

    default_gene = (
        gene_search
        if gene_search in all_genes
        else (gene_order[0] if gene_order else None)
    )
    with gene_tab:
        selected_gene = st.selectbox(
            "Select gene for drill-down",
            options=gene_options,
            index=(
                gene_options.index(default_gene) if default_gene in gene_options else 0
            ),
            key="expr_gene_card",
        )
        if selected_gene:
            gene_df = df_long[df_long["Gene"] == selected_gene]
            box = (
                alt.Chart(gene_df)
                .mark_boxplot()
                .encode(
                    x=alt.X("Subtype_Proxy", title="Subtype"),
                    y=alt.Y("Expression", title=f"{selected_gene} expression"),
                    color=alt.Color("Subtype_Proxy", legend=None),
                    tooltip=["Subtype_Proxy", "Expression"],
                )
                .properties(width=400, title=f"{selected_gene} across subtypes")
            )
            expr_cols = [
                c
                for c in df_samples.columns
                if c not in ["sampleID", "patientID", "Subtype_Proxy", "UMAP1", "UMAP2"]
            ]
            if selected_gene in expr_cols:
                target = df_samples[selected_gene]
                corrs = (
                    df_samples[expr_cols]
                    .corrwith(target)
                    .drop(labels=[selected_gene], errors="ignore")
                    .dropna()
                )
                top_corr = (
                    corrs.abs().sort_values(ascending=False).head(10).reset_index()
                )
                top_corr.columns = ["Gene", "Correlation"]
                corr_chart = (
                    alt.Chart(top_corr)
                    .mark_bar()
                    .encode(
                        y=alt.Y("Gene", sort="-x"),
                        x=alt.X("Correlation", title="|r|"),
                        color=alt.value("#72b7b2"),
                        tooltip=["Gene", "Correlation"],
                    )
                    .properties(
                        width=400, title=f"Top 10 genes correlated with {selected_gene}"
                    )
                )
            else:
                corr_chart = st.info("Correlation not available for this gene.")
            st.altair_chart(box | corr_chart, use_container_width=True)


def page_survival():
    st.header("BRCA Clinical Survival Analysis Dashboard")
    try:
        brca_df = load_and_preprocess_brca("clinical.tsv")
    except Exception:
        st.error("Error: Could not locate clinical.tsv in the specified directory.")
        return
    f1, f2 = st.columns(2)
    with f1:
        age_label = st.selectbox(
            "Define Age Range:",
            ["All BRCA", "Young (0-45)", "Adult (45-65)", "Senior (65+)"],
        )
    with f2:
        stage_opts = sorted([str(x) for x in brca_df["stage"].unique() if pd.notna(x)])
        selected_stage = st.selectbox(
            "Define Cancer Stage:", ["All Stages"] + stage_opts
        )
    filtered_df = brca_df.copy()
    if selected_stage != "All Stages":
        filtered_df = filtered_df[filtered_df["stage"] == selected_stage]
    if "Young" in age_label:
        filtered_df = filtered_df[filtered_df["age"] <= 45]
    elif "Adult" in age_label:
        filtered_df = filtered_df[
            (filtered_df["age"] > 45) & (filtered_df["age"] <= 65)
        ]
    elif "Senior" in age_label:
        filtered_df = filtered_df[filtered_df["age"] > 65]
    st.write("##### Cohort Demographics (Filtered Selection)")
    dist1, dist2 = st.columns(2)
    with dist1:
        age_dist = (
            alt.Chart(filtered_df)
            .mark_bar(color="#4c78a8")
            .encode(
                x=alt.X(
                    "age:Q", bin=alt.Bin(maxbins=20), title="Patient Age at Diagnosis"
                ),
                y=alt.Y("count():Q", title="Patient Frequency"),
                tooltip=["age", "count()"],
            )
            .properties(height=220, title="Age Distribution")
        )
        st.altair_chart(age_dist, use_container_width=True)
    with dist2:
        stage_counts = filtered_df["stage"].value_counts().reset_index()
        stage_counts.columns = ["Stage", "Count"]
        stage_dist = (
            alt.Chart(stage_counts)
            .mark_bar(color="#72b7b2")
            .encode(
                x=alt.X("Count:Q", title="Number of Patients"),
                y=alt.Y("Stage:N", sort="-x", title="AJCC Stage"),
                tooltip=["Stage", "Count"],
            )
            .properties(height=220, title="Stage Prevalence")
        )
        st.altair_chart(stage_dist, use_container_width=True)
    st.divider()
    c1, c2, c3 = st.columns([1, 2.5, 2], gap="large")
    with c1:
        st.write("**Specific Therapy Choice:**")
        therapy_choice = st.radio(
            "Select treatment modality",
            ["Surgery", "Chemotherapy", "Radiation Therapy", "Hormone Therapy"],
            label_visibility="collapsed",
        )
        mapping = {
            "Surgery": "Surgery, NOS",
            "Chemotherapy": "Chemotherapy|Pharmaceutical Therapy, NOS",
            "Radiation Therapy": "Radiation Therapy, NOS|Radiation, External Beam",
            "Hormone Therapy": "Hormone Therapy",
        }
        cohort_data = filtered_df[
            filtered_df["treatment"].str.contains(
                mapping[therapy_choice], case=False, na=False
            )
        ]
        st.metric("Patient Cohort Count", len(cohort_data))
    with c2:
        if len(cohort_data) >= 3:
            st.pyplot(plot_km_brca(cohort_data, therapy_choice))
        else:
            st.warning("Sample size is too small to calculate survival probability.")
    with c3:
        if not cohort_data.empty:
            st.write("**Clinical Category Profile**")
            profile_field = (
                "agent" if cohort_data["agent"].nunique() > 1 else "treatment"
            )
            counts = cohort_data[profile_field].value_counts().nlargest(8).reset_index()
            counts.columns = ["Sub-Category", "Patient Count"]
            donut_viz = (
                alt.Chart(counts)
                .mark_arc(innerRadius=50)
                .encode(
                    theta="Patient Count",
                    color="Sub-Category",
                    tooltip=["Sub-Category", "Patient Count"],
                )
                .properties(height=300)
            )
            st.altair_chart(donut_viz, use_container_width=True)


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
        kmf.fit(df["time"], df["event"], label=f"{therapy_name} Cohort")
        kmf.plot_survival_function(ax=ax)
    ax.set_title(f"Survival Probability: {therapy_name}")
    ax.set_ylabel("Survival Probability")
    ax.set_xlabel("Days since Diagnosis")
    plt.tight_layout(pad=2.0)
    return fig


def page_demographics():
    """
    Global and demographic patterns of breast cancer by stage and demographics.

    Features:
    - Stage filter
    - Age range slider
    - World map showing total BRCA incidence by country
    - Age distribution by race/ethnicity heatmap
    - Stage distribution by demographics stacked bar charts
    """
    st.header("Global and Demographic Patterns of Breast Cancer by Stage and Demographics")
    
    # Load data
    try:
        brca_df = load_and_preprocess_brca("clinical.tsv", drop_missing_time=False)
    except:
        st.error("Error: Could not locate clinical.tsv")
        return
    
    st.subheader("Filters")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        # Stage filter
        stage_opts = sorted([x for x in brca_df["stage"].unique() if pd.notna(x) and x != "Unknown"])
        selected_stage = st.selectbox(
            "STAGE",
            ["All Stages"] + stage_opts
        )
        
        # Normalize by toggle
        normalize_by = st.radio(
            "Normalize by",
            ["Total Cases", "Absolute Counts"],
            horizontal=False
        )
    
    with filter_col2:
        # Filter text box
        filter_text = st.text_input("Filter (race/ethnicity)", "")
        
        # Age range slider
        age_range = st.slider(
            "Age Range",
            int(brca_df["age"].min()),
            int(brca_df["age"].max()),
            (int(brca_df["age"].min()), int(brca_df["age"].max()))
        )
    
    with filter_col3:
        # Include unknown/missing checkbox
        include_unknown = st.checkbox("Include unknown / missing", value=True)
        
        # Patients metric placeholder (will update after filtering)
        patient_metric_placeholder = st.empty()
    
    # --- Apply Filters ---
    filtered_df = brca_df.copy()
    
    # Stage filter
    if selected_stage != "All Stages":
        filtered_df = filtered_df[filtered_df["stage"] == selected_stage]
    
    # Age range filter
    filtered_df = filtered_df[(filtered_df["age"] >= age_range[0]) & (filtered_df["age"] <= age_range[1])]
    
    # Unknown/missing filter across key demographic fields
    if not include_unknown:
        keep_mask = (
            (filtered_df["stage"].notna()) & (filtered_df["stage"] != "Unknown") &
            (filtered_df["race"].notna()) & (filtered_df["race"] != "Unknown") &
            (filtered_df["ethnicity"].notna()) & (filtered_df["ethnicity"] != "Unknown") &
            (filtered_df["site"].notna()) & (filtered_df["site"] != "Unknown") &
            (filtered_df["country"].notna()) & (filtered_df["country"] != "Unknown")
        )
        filtered_df = filtered_df[keep_mask]
    
    # Text filter (searches across race, ethnicity)
    if filter_text:
        mask = (
            filtered_df["race"].astype(str).str.contains(filter_text, case=False, na=False) |
            filtered_df["ethnicity"].astype(str).str.contains(filter_text, case=False, na=False)
        )
        filtered_df = filtered_df[mask]
    
    if filtered_df.empty:
        st.warning("No data matches current filters. Please adjust your selection.")
        return
    
    # Update patient count metric
    with filter_col3:
        patient_metric_placeholder.metric("Patients in View", len(filtered_df))
    
    st.divider()
    st.subheader("World Map: Total BRCA Incidence by Country")
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
        st.caption(
            "📍 Hover over countries to see case counts (derived from clinical.tsv)"
        )
    else:
        st.info("No country data available after applying filters.")
    st.subheader("Age Distribution by Demographics")
    
    # Create age bins for better heatmap visualization
    age_bins = [0, 30, 40, 50, 60, 70, 100]
    age_labels = ["<30", "30-40", "40-50", "50-60", "60-70", "70+"]
    filtered_df["age_group"] = pd.cut(filtered_df["age"], bins=age_bins, labels=age_labels, right=False)
    
    # Age distribution by race
    age_by_race = filtered_df.groupby(["race", "age_group"]).size().reset_index(name="Count")
    if not include_unknown:
        age_by_race = age_by_race[age_by_race["race"] != "Unknown"]
    
    if not age_by_race.empty:
        # Create heatmap showing age distribution across race
        age_race_heatmap = alt.Chart(age_by_race).mark_rect().encode(
            x=alt.X("age_group:O", title="Age Group"),
            y=alt.Y("race:N", title="Race"),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), title="Patient Count"),
            tooltip=["race", "age_group", "Count"]
        ).properties(height=300, width=600)
        
        st.altair_chart(age_race_heatmap, use_container_width=True)
    else:
        st.info("No age/race data available for current filters.")
    st.divider()
    st.subheader("Stage at Diagnosis by Demographics")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**Stage Distribution by Race**")
        
        # Stage by race
        stage_by_race = (
            filtered_df.groupby(["race", "stage"]).size().reset_index(name="Count")
        )
        if not include_unknown:
            stage_by_race = stage_by_race[stage_by_race["race"] != "Unknown"]
            stage_by_race = stage_by_race[stage_by_race["stage"] != "Unknown"]
        if not stage_by_race.empty:
            stage_by_race["Total"] = stage_by_race.groupby("race")["Count"].transform(
                "sum"
            )
            stage_by_race["Percentage"] = (
                100 * stage_by_race["Count"] / stage_by_race["Total"]
            ).round(1)
            stage_race_chart = (
                alt.Chart(stage_by_race)
                .mark_bar()
                .encode(
                    x=alt.X("Percentage:Q", title="Percentage (%)", stack="normalize"),
                    y=alt.Y("race:N", title="Race"),
                    color=alt.Color(
                        "stage:N", title="Stage", scale=alt.Scale(scheme="category10")
                    ),
                    tooltip=["race", "stage", "Count", "Percentage"],
                )
                .properties(height=300)
            )
            st.altair_chart(stage_race_chart, use_container_width=True)
        else:
            st.info("No stage/race data available.")
    with col2:
        st.write("**Stage Distribution by Ethnicity**")
        stage_by_ethnicity = (
            filtered_df.groupby(["ethnicity", "stage"]).size().reset_index(name="Count")
        )
        if not include_unknown:
            stage_by_ethnicity = stage_by_ethnicity[
                stage_by_ethnicity["ethnicity"] != "Unknown"
            ]
            stage_by_ethnicity = stage_by_ethnicity[
                stage_by_ethnicity["stage"] != "Unknown"
            ]
        if not stage_by_ethnicity.empty:
            stage_by_ethnicity["Total"] = stage_by_ethnicity.groupby("ethnicity")[
                "Count"
            ].transform("sum")
            stage_by_ethnicity["Percentage"] = (
                100 * stage_by_ethnicity["Count"] / stage_by_ethnicity["Total"]
            ).round(1)
            stage_ethnicity_chart = (
                alt.Chart(stage_by_ethnicity)
                .mark_bar()
                .encode(
                    x=alt.X("Percentage:Q", title="Percentage (%)", stack="normalize"),
                    y=alt.Y("ethnicity:N", title="Ethnicity"),
                    color=alt.Color(
                        "stage:N", title="Stage", scale=alt.Scale(scheme="category10")
                    ),
                    tooltip=["ethnicity", "stage", "Count", "Percentage"],
                )
                .properties(height=300)
            )
            st.altair_chart(stage_ethnicity_chart, use_container_width=True)
        else:
            st.info("No stage/ethnicity data available.")
    st.divider()
    
    st.divider()
    
    # --- Summary Statistics ---
    st.subheader("Summary Statistics")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("Total Patients", len(filtered_df))
    with summary_col2:
        st.metric(
            "Stages Represented",
            filtered_df[filtered_df["stage"] != "Unknown"]["stage"].nunique(),
        )
    with summary_col3:
        st.metric("Avg Age at Diagnosis", f"{filtered_df['age'].mean():.1f}")


def page_anatomy():
    st.header("Tumor anatomical location co-occurrence and stage")
    active_label = st.selectbox("Select organ to analyze", ["None"] + organs, index=0)
    active_organ = None if active_label == "None" else active_label
    col1, col2, col3 = st.columns([0.45, 0.25, 0.8])
    with col1:
        st.subheader("Tumor site prevalence and co-occurrence")
        fig_anatomy = make_figure(active_organ=active_organ)
        st.plotly_chart(fig_anatomy, use_container_width=False)
    with col2:
        st.markdown("###  ")
        st.image("data/legend.png", use_container_width=True)
    with col3:
        st.subheader("Tumor Stage Distribution")
        fig_stage = make_stage_chart(active_organ)
        st.plotly_chart(fig_stage, use_container_width=True)
    st.subheader("Visualize co-occurrence counts of tumor sites")
    fig_upset = make_upset_plot(active_organ)
    st.plotly_chart(fig_upset, use_container_width=True)


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to", ["Gene Expression", "Survival", "Demographics", "Anatomy"]
    )
    if page == "Gene Expression":
        page_expression()
    elif page == "Survival":
        page_survival()
    elif page == "Demographics":
        page_demographics()
    elif page == "Anatomy":
        page_anatomy()


if __name__ == "__main__":
    main()
