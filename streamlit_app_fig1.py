import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

st.set_page_config(layout="wide", page_title="TCGA-BRCA Explorer")


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


st.title("🧬 TCGA-BRCA Gene Expression Explorer")

st.sidebar.header("Controls")
cluster_method = st.sidebar.selectbox(
    "Clustering linkage method",
    ["ward", "average", "complete", "single"],
    index=0,
    help="Change how hierarchical clustering groups genes/samples",
)
distance_metric = st.sidebar.selectbox(
    "Distance metric",
    ["euclidean", "cityblock", "cosine"],
    index=0,
    help="Metric for computing pairwise distances",
)
heatmap_filter_mode = st.sidebar.radio(
    "Heatmap selection mode",
    ["Brush from scatter", "Click bar -> highlight"],
    index=0,
)

scatter_mode = st.sidebar.radio(
    "Scatter mode",
    ["UMAP", "Manual gene axes"],
    index=0,
    help="Use UMAP embedding or pick two genes manually",
)

df_samples, df_long = load_and_process_data()

if df_samples is not None:
    all_genes = [g for g in df_long["Gene"].unique()]
    gene_options = sorted(all_genes) if all_genes else ["N/A"]
    subtype_options = sorted(df_samples["Subtype_Proxy"].dropna().unique())

    select_all_subtypes = st.sidebar.checkbox("Select all subtypes", value=True)
    selected_subtypes = st.sidebar.multiselect(
        "Filter by subtype",
        subtype_options,
        default=subtype_options,
    )
    if select_all_subtypes:
        selected_subtypes = subtype_options

    select_all_genes = st.sidebar.checkbox("Select all genes (shown)", value=False)
    gene_filter = st.sidebar.multiselect(
        "Limit heatmap to genes",
        options=gene_options,
        default=gene_options[: min(20, len(gene_options))],
        help="Choose subset to speed rendering; gene search always adds a match",
    )
    if select_all_genes:
        gene_filter = gene_options

    gene_search = st.sidebar.text_input("Find Gene (exact match)", "")

    g1_manual = st.sidebar.selectbox(
        "Scatter X gene",
        options=gene_options,
        index=0,
    )
    g2_manual = st.sidebar.selectbox(
        "Scatter Y gene",
        options=gene_options,
        index=1 if len(gene_options) > 1 else 0,
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
            df_samples["UMAP1"], df_samples["UMAP2"] = (
                embedding[:, 0],
                embedding[:, 1],
            )
            umap_ready = True
        except Exception:
            umap_ready = False

if df_samples is not None:
    matrix = df_long.pivot(index="Gene", columns="sampleID", values="Expression")

    if matrix.empty:
        st.error(
            "❌ No overlapping expression/clinical samples available for visualization."
        )
        st.stop()

    gene_order = matrix.index.tolist()
    if "gene_search" in locals() and gene_search in gene_order:
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

    if len(gene_order) >= 2:
        if scatter_mode == "Manual gene axes":
            g1, g2 = g1_manual, g2_manual
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
                .properties(title="Sample Similarity (UMAP)", width=420, height=320)
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
                    title="Sample Similarity (Gene-Gene)", width=420, height=320
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
        .properties(title="Subtype counts", width=220, height=180)
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
            y=alt.Y("Gene", sort="-x"),
            x=alt.X("Expression", title="Variance"),
            color=alt.value("#4c78a8"),
            tooltip=["Gene", alt.Tooltip("Expression", title="Variance")],
        )
        .add_params(gene_bar_selection)
        .properties(title="Top genes by variance", width=260, height=400)
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
        .properties(title="Top 50 Variable Genes", width=500)
    )

    if heatmap_filter_mode == "Brush from scatter" and brush is not None:
        heatmap = heatmap.transform_filter(brush)

    if heatmap_filter_mode == "Click bar -> highlight":
        heatmap = heatmap.transform_filter(bar_selection)

    heatmap = heatmap.transform_filter(gene_bar_selection)

    left_col = scatter
    if scatter is not None:
        left_col = (scatter & subtype_bar).resolve_scale(color="independent")
    else:
        left_col = subtype_bar

    left_col = left_col & top_gene_bar

    st.altair_chart(left_col | heatmap, use_container_width=True)

    st.markdown("### 🔍 Gene Card")
    default_gene = (
        gene_search
        if gene_search in all_genes
        else (gene_order[0] if gene_order else None)
    )
    selected_gene = st.selectbox(
        "Select gene for drill-down",
        options=gene_options,
        index=(gene_options.index(default_gene) if default_gene in gene_options else 0),
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
            top_corr = corrs.abs().sort_values(ascending=False).head(10).reset_index()
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
