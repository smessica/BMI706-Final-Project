# TCGA BRCA Dashboard

This is a comprehensive data visualization dashboard for exploring TCGA Breast Invasive Carcinoma (BRCA) data. The application is built using Streamlit and integrates various visualization libraries including Altair, Plotly, and Matplotlib to provide deep insights into gene expression, clinical survival, demographics, and anatomical tumor distribution.

## Features

The dashboard consists of four main sections:

### 1. Gene Expression Explorer
Explore gene expression patterns and sample similarities.
- **Interactive Scatter Plot**: Visualize sample similarity using UMAP or Gene-Gene expression axes. Supports brushing for selection.
- **Heatmap**: View expression levels of the top 50 variable genes.
- **Subtype Analysis**: Bar charts showing sample counts by subtype.
- **Gene Drill-Down**: Detailed view for specific genes including expression boxplots across subtypes and correlation analysis with other genes.
- **Clustering Options**: Customizable linkage methods (ward, average, complete, single) and distance metrics (euclidean, cityblock, cosine).

### 2. Survival Analysis
Analyze clinical survival data based on various cohorts.
- **Cohort Filtering**: Filter patients by age group (Young, Adult, Senior) and cancer stage.
- **Kaplan-Meier Plots**: Generate survival curves based on therapy choices (Surgery, Chemotherapy, Radiation, Hormone Therapy).
- **Demographic Distributions**: Visualize age and stage distributions for the selected cohort.
- **Treatment Profiling**: Donut charts showing the distribution of specific therapeutic agents.

### 3. Demographics
Investigate global and demographic patterns of breast cancer.
- **Global Incidence**: Interactive choropleth map showing BRCA cases by country.
- **Demographic Heatmaps**: Age distribution by race.
- **Stage Distribution**: Stacked bar charts showing cancer stage distribution across different races and ethnicities.
- **Advanced Filtering**: Filter by stage, age range, and text search (race, ethnicity, site).
- **Summary Statistics**: Key metrics including total patients, unique sites, and average age.

### 4. Anatomy
Visualize tumor anatomical location, prevalence, and co-occurrence.
- **Anatomical Map**: Interactive SVG-based visualization of tumor sites (breast quadrants, liver, lung, bone, etc.) colored by prevalence.
- **Co-occurrence Analysis**: Visualize how often tumors occur in multiple sites simultaneously.
- **Stage Distribution by Organ**: Bar chart showing tumor stage distribution for selected anatomical sites.
- **Upset Plot**: Analyze complex co-occurrence patterns of tumor sites.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd BMI706-Final-Project
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If `requirements.txt` is not provided, the main dependencies are `streamlit`, `pandas`, `numpy`, `altair`, `plotly`, `matplotlib`, `lifelines`, `scipy`, and `umap-learn`.*

3. Ensure the data files are in the correct location:
   - `clinical.tsv`: Clinical data file.
   - `expression.txt`: Gene expression data file.
   - `data/anatomy.svg`: SVG template for the anatomy visualization.

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The dashboard will open in your default web browser. Use the sidebar to navigate between the different analysis modules.

## Data Sources

- **TCGA BRCA**: The data is sourced from The Cancer Genome Atlas (TCGA) Breast Invasive Carcinoma project.
- **Clinical Data**: Contains patient demographics, diagnoses, and treatment information.
- **Expression Data**: Contains gene expression levels for samples.

## License

[Insert License Information Here] 
