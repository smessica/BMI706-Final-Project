import streamlit as st
import plotly.graph_objects as go
import xml.etree.ElementTree as ET
import re
from collections import defaultdict, Counter
from base64 import b64encode
import pandas as pd
import math
from plotly.subplots import make_subplots

# Set page to wide layout
st.set_page_config(layout="wide")


# Read the anatomic SVG template
# SVG files downloaded from wikipedia commons and svgrepo -> modified manually in illustrator
with open("data/anatomy.svg", "rb") as f:
    SVG_TEMPLATE_BYTES = f.read()


# References: 
# https://stackoverflow.com/questions/22252472/how-can-i-change-the-color-of-an-svg-element
# https://stackoverflow.com/questions/35931133/how-to-replace-attributes-of-each-element-in-an-xml-svg-file-and-then-save-it-of
def apply_color_to_element(elem, hex_color):
    SHAPE_TAGS = ("path", "rect", "circle", "ellipse", "polygon", "polyline")
    for node in elem.iter():
        tag = node.tag.split("}")[-1]
        if tag in SHAPE_TAGS:
            style = node.get("style", "")
            style = re.sub(r"fill:[^;]+;?", "", style)
            style = style.strip().rstrip(";")
            if style:
                style += ";"
            style += f"fill:{hex_color}"
            node.set("style", style)
            node.set("fill", hex_color)
def generate_colored_svg_data_uri(prevalence_dict, active_organ=None, co_occurrence_dict=None):
    root = ET.fromstring(SVG_TEMPLATE_BYTES)

    # Map Breast quadrants
    quadrant_mapping = {
        # Left breast (4 quadrants)
        "left_upper_inner": "quadrant_path_1",
        "left_lower_inner": "quadrant_path_2",
        "left_lower_outer": "quadrant_path_3",
        "left_upper_outer": "quadrant_path_4",
        # Right breast (4 quadrants)
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

    # Map organs
    organ_to_group_id = {
        "abdomen": "group_abdomen",
        "bone": "group_bone",
        "skin": "group_skin",
        "lymph_node": "group_lymph_node",
        "ovary": "group_ovary",
        "adrenal_gland": "group_adrenal_gland",
    }

    # lung & liver mapping
    for i, child in enumerate(root):
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
    
    # Map regular organs to their group IDs
    for organ_name, group_id in organ_to_group_id.items():
        name_to_ids[organ_name].append(group_id)
    
    # Map breast quadrants to their path IDs
    for qname, path_id in quadrant_mapping.items():
        name_to_ids[qname].append(path_id)

    # Color each organ based on context
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


# organ positions
organ_positions = {
    # Breast quadrants
    # Left breast quadrants
    "left_upper_inner": (405, 270),
    "left_lower_inner": (360, 345),
    "left_lower_outer": (315, 355),
    "left_upper_outer": (312, 270),
    # Right breast quadrants
    "right_upper_inner": (495, 270),
    "right_upper_outer": (585, 270),
    "right_lower_outer": (583, 345),
    "right_lower_inner": (540, 345),
    # Abdomen
    "abdomen":          (450, 490),
    "lung":             (100, 350),
    "liver":            (100, 540),
    "adrenal_gland":    (83, 755),
    "bone":             (770, 370),
    "skin":             (770, 540),
    "lymph_node":       (775, 745),
    "ovary":            (445, 720),
}
organs = list(organ_positions.keys())

# Site to organ mapping
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
BREAST_QUADRANTS = {"left_upper_outer", "left_upper_inner", "left_lower_outer", "left_lower_inner",
                    "right_upper_outer", "right_upper_inner", "right_lower_outer", "right_lower_inner"}

def map_site_to_organ(site):
    return SITE_TO_ORGAN.get(site)

@st.cache_data
def load_and_process_clinical_data():
    """Load clinical data and calculate prevalence and co-occurrence."""
    # Load clinical data
    clinical = pd.read_csv('data/clinical.tsv', sep='\t', low_memory=False)
    clinical.replace("'--", pd.NA, inplace=True)
    
    # Create list of (case_id, organ_list) tuples
    case_organs = []
    for idx, row in clinical.iterrows():
        sites_str = row["diagnoses.sites_of_involvement"]
        if pd.notna(sites_str):
            sites = [s.strip() for s in sites_str.split('|')]
            organs = [map_site_to_organ(s) for s in sites]
            organs = [o for o in organs if o is not None]
            if organs:
                case_organs.append((row['cases.case_id'], organs))
    
    # Calculate prevalence
    all_organs_list = [organ for case_id, organs in case_organs for organ in organs]
    organ_counts = pd.Series(all_organs_list).value_counts()
    total_cases = len(case_organs)
    
    prevalence = {organ: organ_counts[organ] / total_cases for organ in organ_counts.index}
    counts = {organ: organ_counts[organ] for organ in organ_counts.index}
    
    # Calculate co-occurrence
    all_organs_set = set(prevalence.keys())
    co_occurrence = {}
    
    for organ_a in all_organs_set:
        co_occurrence[organ_a] = {}
        cases_with_a = sum(1 for case_id, organs in case_organs if organ_a in organs)
        
        for organ_b in all_organs_set:
            if organ_a != organ_b:
                cases_with_both = sum(1 for case_id, organs in case_organs 
                                      if organ_a in organs and organ_b in organs)
                co_occurrence[organ_a][organ_b] = cases_with_both / cases_with_a if cases_with_a > 0 else 0.0
    
    return clinical, prevalence, co_occurrence, counts

clinical, prevalence, co_occurrence, counts = load_and_process_clinical_data()


# Tumor stage data prepraration
def get_stage_data(selected_organ=None):
    if selected_organ is None:
        # Return overall stage distribution
        stage_counts = clinical["diagnoses.ajcc_pathologic_stage"].value_counts()
        return stage_counts
    else:
        # Filter by selected organ
        filtered_cases = []
        for idx, row in clinical.iterrows():
            sites_str = row["diagnoses.sites_of_involvement"]
            if pd.notna(sites_str):
                sites = [s.strip() for s in sites_str.split('|')]
                organs = [map_site_to_organ(s) for s in sites]
                if selected_organ in organs:
                    filtered_cases.append({"diagnoses.ajcc_pathologic_stage": row["diagnoses.ajcc_pathologic_stage"]})
        
        if not filtered_cases:
            return pd.Series(dtype=int)
        
        filtered_df = pd.DataFrame(filtered_cases)
        stage_counts = filtered_df["diagnoses.ajcc_pathologic_stage"].value_counts()
        return stage_counts

# upset plot data preparation
def get_organ_combinations():
    combinations = []
    for idx, row in clinical.iterrows():
        sites_str = row["diagnoses.sites_of_involvement"]
        if pd.notna(sites_str):
            sites = [s.strip() for s in sites_str.split('|')]
            organs = [map_site_to_organ(s) for s in sites]
            organs = [o for o in organs if o is not None]
            
            if organs:
                # Store as a frozenset for easy comparison
                combinations.append(frozenset(organs))
    
    # Count occurrences of each combination
    combo_counts = Counter(combinations)
    return combo_counts


def make_upset_plot(selected_organ=None):
    combo_counts = get_organ_combinations()
    
    # Get top 20
    top_combos = sorted(combo_counts.items(), key=lambda x: -x[1])[:20]
    
    # prepare organ list
    all_organs_in_combos = set()
    for combo, _ in top_combos:
        all_organs_in_combos.update(combo)
    organs_list = sorted(all_organs_in_combos)
    n_combos = len(top_combos)
    n_organs = len(organs_list)
    
    # highlight selected
    highlight_indices = []
    if selected_organ:
        for i, (combo, _) in enumerate(top_combos):
            if selected_organ in combo:
                highlight_indices.append(i)
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.05,
        subplot_titles=("", ""),
        specs=[[{"type": "bar"}], [{"type": "scatter"}]]
    )
    
    # Top panel: count
    counts = [count for _, count in top_combos]
    bar_colors = ['#ff6b6b' if i in highlight_indices else 'steelblue' for i in range(n_combos)]
    combo_hover_texts = []
    for combo, count in top_combos:
        organs_display = '<br>'.join([o.replace('_', ' ').title() for o in sorted(combo)])
        combo_hover_texts.append(f"<b>Combination ({len(combo)} organs):</b><br>{organs_display}<br><b>Count:</b> {count}")
    
    fig.add_trace(go.Bar(
        x=list(range(n_combos)),
        y=counts,
        marker=dict(color=bar_colors),
        showlegend=False,
        text=counts,
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=combo_hover_texts,
        name="Count"
    ), row=1, col=1)
    
    # Bottom panel: organ combination
    for j, (combo, count) in enumerate(top_combos):
        organs_in_combo = [organs_list.index(o) for o in combo if o in organs_list]
        if len(organs_in_combo) > 1:
            organs_in_combo.sort()
            for i in range(len(organs_in_combo) - 1):
                is_highlighted = j in highlight_indices
                fig.add_trace(go.Scatter(
                    x=[j, j],
                    y=[organs_in_combo[i], organs_in_combo[i+1]],
                    mode='lines',
                    line=dict(
                        color='#ff6b6b' if is_highlighted else '#4a4a4a',
                        width=3 if is_highlighted else 2
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=2, col=1)
    for i, organ in enumerate(organs_list):
        x_coords = []
        y_coords = []
        hover_texts = []
        for j, (combo, count) in enumerate(top_combos):
            if organ in combo:
                x_coords.append(j)
                y_coords.append(i)
                other_organs = [o.replace('_', ' ').title() for o in sorted(combo) if o != organ]
                hover_text = f"<b>{organ.replace('_', ' ').title()}</b><br>Co-occurs with:<br>" + '<br>'.join(other_organs) + f"<br><b>Total count:</b> {count}"
                hover_texts.append(hover_text)
        
        # Highlight selected organ
        is_selected = (organ == selected_organ)
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=12,
                color='#ff6b6b' if is_selected else '#4a4a4a',
                symbol='circle',
                line=dict(width=1, color='white')
            ),
            name=organ,
            showlegend=False,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts
        ), row=2, col=1)
    
    # Update layout
    fig.update_xaxes(
        title="",
        range=[-0.5, n_combos - 0.5],
        showticklabels=False,
        row=1, col=1
    )
    fig.update_yaxes(
        title="Count",
        row=1, col=1
    )
    
    fig.update_xaxes(
        title="",
        range=[-0.5, n_combos - 0.5],
        showticklabels=False,
        row=2, col=1
    )
    fig.update_yaxes(
        title="Organs",
        ticktext=[o.replace('_', ' ').title() for o in organs_list],
        tickvals=list(range(n_organs)),
        range=[-0.5, n_organs - 0.5],
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=150, r=40, t=60, b=60),
        title="Upset Plot (Top 20)" + 
              (f" - Highlighting: {selected_organ.replace('_', ' ').title()}" if selected_organ else ""),
        hovermode='closest',
        showlegend=False
    )
    
    return fig

def prevalence_to_hex(p, organ_name, is_active=False, co_occurrence_value=None):
    # custom color scale
    p = max(0.0, min(1.0, p))
    
    # Active organ
    if is_active:
        return "#0066cc"
    
    # Color co-occurance 
    if co_occurrence_value is not None:
        co_val = max(0.0, min(1.0, co_occurrence_value))
        # scale for better differentiation
        if co_val > 0:
            co_val_scaled = math.log10(1 + co_val * 99) / 2
        else:
            co_val_scaled = 0
        if organ_name in BREAST_QUADRANTS:
            # Red color scale for breast quadrants
            r = int(255 + co_val_scaled * (150 - 255))
            g = int(230 + co_val_scaled * (0 - 230)) 
            b = int(230 + co_val_scaled * (0 - 230))
        else:
            # Green color scale for other organs
            r = int(230 + co_val_scaled * (0 - 230))
            g = int(255 + co_val_scaled * (150 - 255))
            b = int(230 + co_val_scaled * (0 - 230))
        return f"#{r:02x}{g:02x}{b:02x}"
    
    # Color by prevalence if no organ selected
    # Scale for better differentiation
    if p > 0:
        p_scaled = math.log10(1 + p * 99) / 2
    else:
        p_scaled = 0
    
    if organ_name in BREAST_QUADRANTS:
        # Red color scale for breast quadrants
        r = int(255 + p_scaled * (150 - 255))
        g = int(230 + p_scaled * (0 - 230))
        b = int(230 + p_scaled * (0 - 230))
    else:
        # Green color scale for other organs
        r = int(230 + p_scaled * (0 - 230))
        g = int(255 + p_scaled * (150 - 255))
        b = int(230 + p_scaled * (0 - 230))
    
    return f"#{r:02x}{g:02x}{b:02x}"

# References
# https://plotly.com/python/images/
# https://plotly.com/python/shapes/
# https://plotly.com/python/text-and-annotations/
def make_figure(active_organ=None):
    # Color SVG
    bg_src = generate_colored_svg_data_uri(prevalence, active_organ=active_organ, co_occurrence_dict=co_occurrence)

    # Scale positions
    scale_x = 500 / 859
    scale_y = 500 / 840.6
    xs, ys, texts, customdata = [], [], [], []
    for organ in organs:
        x, y = organ_positions[organ]
        xs.append(x * scale_x)
        ys.append(y * scale_y)
        texts.append(organ)
        
        # hover text
        organ_display = organ.replace('_', ' ').title()
        count = counts.get(organ, 0)
        hover_text = f"<b>{organ_display}</b><br>Count: {count}<br>"
        
        if active_organ and organ == active_organ:
            hover_text += "<b>Selected organ</b>"
        elif active_organ and co_occurrence.get(active_organ, {}).get(organ):
            co_val = co_occurrence[active_organ][organ]
            hover_text += f"Co-occurrence with {active_organ.replace('_', ' ').title()}: {co_val:.2%}"
        
        customdata.append(hover_text)

    scatter = go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        text=texts,
        textposition="top center",
        marker=dict(
            size=60,
            color="rgba(0,0,0,0)",
            line=dict(width=0),
        ),
        hovertemplate='%{customdata}<extra></extra>',
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

    # add svg as background
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

    # add arrows for co-occurrence
    shapes = []
    annotations = []
    if active_organ is not None and active_organ in co_occurrence:
        # Scaling
        scale_x = 500 / 859
        scale_y = 500 / 840.6
        
        x0, y0 = organ_positions[active_organ]
        x0_scaled = x0 * scale_x
        y0_scaled = y0 * scale_y
        
        targets = co_occurrence[active_organ]
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
                
                # Add arrows
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


def make_stage_chart(selected_organ=None):
    stage_data = get_stage_data(selected_organ)
    # Sort stages
    stage_order = ['Stage 0', 'Stage 0is', 'Stage I', 'Stage IA', 'Stage IB', 'Stage II', 
                   'Stage IIA', 'Stage IIB', 'Stage III', 'Stage IIIA', 'Stage IIIB', 
                   'Stage IIIC', 'Stage IV', 'Stage X']
    ordered_stages = [s for s in stage_order if s in stage_data.index]
    ordered_counts = [stage_data[s] for s in ordered_stages]
    
    # hover text
    total_cases = sum(ordered_counts)
    hover_texts = []
    for stage, count in zip(ordered_stages, ordered_counts):
        percentage = (count / total_cases * 100) if total_cases > 0 else 0
        hover_text = f"<b>{stage}</b><br>Cases: {count}<br>Percentage: {percentage:.1f}%"
        if selected_organ:
            hover_text += f"<br>For: {selected_organ.replace('_', ' ').title()}"
        hover_texts.append(hover_text)
    
    fig = go.Figure(data=[
        go.Bar(
            x=ordered_stages,
            y=ordered_counts,
            marker=dict(color='steelblue'),
            text=ordered_counts,
            textposition='outside',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts
        )
    ])
    
    title_text = "Tumor Stage Distribution"
    if selected_organ:
        organ_display = selected_organ.replace('_', ' ').title()
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


# Main UI
st.title("Analyze tumor anatomical location's co-occurrence, metastatic spread, and tumor stage")

# Sidebar
with st.sidebar:
    st.header("Settings")
    active_label = st.selectbox(
        "Select organ to analyze:",
        ["None"] + organs,
        index=0,
    )
    st.info("Select an organ to view its co-occurrence patterns with other tumor sites and its tumor stage distribution.")

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
