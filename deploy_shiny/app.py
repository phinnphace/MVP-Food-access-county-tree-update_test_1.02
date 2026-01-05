from shiny import App, ui, render, reactive
import folium
import geopandas as gpd
import pandas as pd
import mapclassify
import branca.colormap as cm
import sys
import numpy as np
import altair as alt  # Charting
from shinywidgets import output_widget, render_altair # Interactive Widgets

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Custom Color Palettes (Aligned with Master Config + Extras)
# Custom Pastel Palette for Internet (5 classes)
PASTEL_COLORS = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6']

# Manual Bin Overrides
MANUAL_BINS = {
    "C_Transit_B": [0, 1, 2, 3, 4, 5],
    "pct_no_internet": [0, 10, 20, 30, 40, 50, 60]
}

# Mapping Variables to Branca Colormaps
def get_layer_colormap(var_id):
    # Maps variable ID to Branca LinearColormap (or list of colors)
    # Using specific cm.linear attributes as provided in user script
    mapping = {
        "V_final_Weighted":   cm.linear.PuBuGn_09,
        "V_final_Unweighted": cm.linear.Oranges_09,
        "C_Transit_B":        cm.linear.Reds_09,
        "C_Vehicle_A":        cm.linear.Purples_09,
        "C_Economic":         cm.linear.Greens_09,
        "C_Internet_C":       cm.LinearColormap(colors=PASTEL_COLORS),
        "pct_no_internet":    cm.LinearColormap(colors=PASTEL_COLORS),
        "C_Roads_D":          cm.linear.YlOrBr_09,
        "C_Geographic":       cm.linear.PuRd_09,
        "LILA_Flag":          cm.LinearColormap(colors=['#f0f0f0', '#d6604d'], vmin=0, vmax=1) # Grey to Red
    }
    return mapping.get(var_id, cm.linear.YlGnBu_09)

VAR_CONFIG = {
    "Assessment Scores": [
        {"id": "V_final_Weighted", "label": "Final Weighted Score", "direction": "High Value = High Vulnerability"},
        {"id": "V_final_Unweighted", "label": "Final Unweighted Score", "direction": "High Value = High Vulnerability"}
    ],
    "Composite Scores": [
        {"id": "C_Economic", "label": "Economic Vulnerability", "direction": "High Score = High Vulnerability"},
        {"id": "C_Geographic", "label": "Geographic Vulnerability", "direction": "High Score = High Vulnerability"},
        {"id": "TVS", "label": "Transportation Vulnerability Score", "direction": "High Score = High Vulnerability"}
    ],
    "Systemic Transit Impact": [
        {"id": "C_Vehicle_A", "label": "No Vehicle % HH", "direction": "Darker = Less Access (High % No Vehicle)"},
        {"id": "C_Transit_B", "label": "Transit Stops per Capita", "direction": "Darker = High Density (Or Desert Penalty)"},
        {"id": "C_Internet_C", "label": "% No Internet Access", "direction": "Darker = Higher % No Internet"},
        {"id": "C_Roads_D", "label": "Walkability", "direction": "Darker = Higher Walkability Score"}
    ]
}

# ==============================================================================
# DATA LOADING (SIMPLIFIED)
# ==============================================================================

def load_data():
    try:
        print("Loading final_app_data.geojson...", flush=True)
        gdf = gpd.read_file("final_app_data.geojson")
        
        # Ensure correct projection for web maps
        if gdf.crs is not None and gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
            
        return gdf
    except Exception as e:
        print(f"CRITICAL ERROR: {e}", file=sys.stderr)
        return gpd.GeoDataFrame()

# Global Load
full_data = load_data()

# Pre-calculate Choices for UI (Prevents empty initial state)
GEOID_CHOICES = {}
if not full_data.empty:
    GEOID_CHOICES = {
        row['GEOID']: f"Tract {row.get('TRACTCE20', 'Unknown')}" 
        for _, row in full_data.iterrows()
    }
# Sort by label
GEOID_CHOICES = dict(sorted(GEOID_CHOICES.items(), key=lambda item: item[1]))

# Load specific grocery points if available (Optional)
try:
    grocery_gdf = gpd.read_file("groceries.geojson")
    if grocery_gdf.crs != "EPSG:4326":
        grocery_gdf = grocery_gdf.to_crs("EPSG:4326")
except Exception:
    grocery_gdf = None

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def create_jenks_colormap(values, var_id, caption):
    # 1. Get Base Colormap from User Config
    base_cmap = get_layer_colormap(var_id)
    
    # Clean NaN values
    cleaned_vals = values.dropna().values
    
    # 2. Determine Bounds (Bins)
    bounds = []
    
    # A. Manual Override
    if var_id in MANUAL_BINS:
        bounds = MANUAL_BINS[var_id]
    else:
        # B. Jenks / Natural Breaks
        if len(np.unique(cleaned_vals)) < 2:
            val = np.mean(cleaned_vals) if len(cleaned_vals) > 0 else 0
            if isinstance(base_cmap, cm.LinearColormap):
                 s = base_cmap.scale(val-0.1, val+0.1).to_step(index=[val-0.1, val+0.1])
                 s.caption = caption
                 return s
            return base_cmap

        try:
            nb = mapclassify.NaturalBreaks(cleaned_vals, k=5)
            # Ensure unique bounds
            bounds = [cleaned_vals.min()] + list(nb.bins)
            bounds = sorted(list(set(bounds)))
        except Exception:
            bounds = [cleaned_vals.min(), cleaned_vals.max()]

    # 3. Apply Bounds to Colormap (Step Colormap)
    if len(bounds) > 1:
        s = base_cmap.to_step(index=bounds)
        s.caption = caption
        return s
    
    return base_cmap.scale(cleaned_vals.min(), cleaned_vals.max())

# ==============================================================================
# UI
# ==============================================================================

app_ui = ui.page_fluid(
    # --- 0. HEAD & STYLES ---
    ui.head_content(
        # MathJax for LaTeX support
        ui.tags.script(src="https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"),
        
        # CSS Tricks
        ui.tags.style("""
            /* Hides the radio button and label for the '_none_' option */
            input[value="_none_"], input[value="_none_"] + span { display: none !important; }
            .radio:has(input[value="_none_"]) { display: none !important; }
            div.radio > label > input[value="_none_"] { display: none !important; }
        """)
    ),

    # --- 1. TITLE & INTRO ---
    # --- 1. TITLE & INTRO ---
    ui.div(
        ui.h3("Seeing through the Map: A Static Test of Classification, Measurement, and Proxy Logic", style="margin-top: 10px; margin-bottom: 0px; font-weight: bold;"),
    ),
    ui.markdown("""
    ### An Invitation to Critique
    This dashboard is not a statement of fact, but a **proposal for measurement**. 
    Standard USDA maps often erase local nuance. This model attempts to correct that by implementing **Contextual Weighting (RUCA)**, and accounting for systemic infrastructure gaps. 
    
    This dashboard demonstrates an **Equity-Centered Scoring Model** where context is used to weight components, correcting for the flaws of 'one-size-fits-all' vulnerability indices. 
    The model incorporates **Composite Economic** and **Systemic Transit Penalty** factors.
    
    The model is far from a benchmark and it is still "better" than a binary measurement that relies on the false assumption that proximity to a resource equals access. 

    **How to use this tool:**
    1.  **Deconstruct:** Use the sidebar to view the raw components (like Transit or Internet).
    2.  **Compare:** Look at how the *Unweighted* score differs from the *Weighted* score.
    3.  **Ask:** Does this classification logic hold up? Where does the proxy logic fail?
    """),
    ui.hr(),

    # --- 2. MAIN MAP (SECTION 1) ---
    ui.h3("1. Final Contextual Vulnerability Score (V_final_Weighted)"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Vulnerability Layers"),
            
            ui.h5("Assessment Scores"),
            ui.div(
                ui.input_radio_buttons("rb_assess", None, {**{"_none_": "Click to clear"}, **{x["id"]: x["label"] for x in VAR_CONFIG["Assessment Scores"]}}, selected=None),
                style="margin-left: 10px;"
            ),
            
            ui.h5("Composite Scores"),
            ui.div(
                ui.input_radio_buttons("rb_comp", None, {**{"_none_": "Click to clear"}, **{x["id"]: x["label"] for x in VAR_CONFIG["Composite Scores"]}}, selected=None),
                style="margin-left: 10px;"
            ),
            
            ui.h5("Systemic Transit Impact"),
            ui.div(
                ui.input_radio_buttons("rb_systemic", None, {**{"_none_": "Click to clear"}, **{x["id"]: x["label"] for x in VAR_CONFIG["Systemic Transit Impact"]}}, selected=None),
                style="margin-left: 10px;"
            ),
            
            ui.hr(),
            ui.input_action_button("btn_clear", "Clear Map", class_="btn-danger", style="width: 100%;"),
            ui.markdown("**Controls:**\nSelect a layer to visualize. Click 'Clear Map' to reset.")
        ),
        ui.card(
            ui.output_ui("main_map_ui"),
        )
    ),
    
    # --- Contextual Text Block (Added per user request) ---
    ui.markdown("""
    #### Presumptive norms lead to disproportionate effects.
    Take the approach the USDA uses for the “food desert” flag (LILA tracts) in the <a href="https://www.ers.usda.gov/data-products/food-access-research-atlas/" target="_blank">Food Access Research Atlas</a>:
    
    > “A tract is classified as ‘low-income, low-access’ (LILA) if it meets both low-income criteria (poverty rate ≥20% or median family income ≤80% of state/metro median) and low-access criteria (at least 500 people or 33% of the population more than 1 mile [urban] or 10 miles [rural] from the nearest supermarket, supercenter, or large grocery), as defined by <a href="https://www.ers.usda.gov" target="_blank">USDA ERS (2023)</a>.”

    In summation; resource proximity. How close are you or I to the resource states how accessible it is to us. There are refinements to this where they add in the vehicle by household percentage. 
    As a (recent) non-driver due to disability in a small city with sub-optimal transit I know first hand how inaccurate these measures are. 

    Decatur County, GA was randomly selected for this static model and is ~ 35 miles wide; given the above definition, the grocery points that are placed on the map from <a href="https://www.openstreetmap.org" target="_blank">OpenStreetMap</a> mean that almost everyone falls within the range of grocery store access. 
    **Accessibility is far more complicated than that.**
    """),
    # --- 3. COMPARE MODELS (SECTION 2 - NEW) ---
    ui.hr(),
    ui.h3("2. Context vs. Flattening: Comparison of Scoring Models"),
    ui.layout_columns(
        ui.card(
            ui.h4("Select Tract for Status:", style="margin-top: 0px; font-size: 16px;"),
            ui.input_select("geo_compare", None, GEOID_CHOICES)
        ),
        ui.card(
            ui.output_ui("lila_callout"),
            style="display: flex; align-items: center; justify-content: center; text-align: center;"
        ),
        col_widths=(4, 8)
    ),
    ui.markdown("""
    The comparison shows the difference between the **Unweighted Score** (simple additive) 
    and the **Weighted Score** (using Transport Vulnerability Score with RUCA context weights).
    """),
    ui.card(
        output_widget("chart_comp")
    ),

    # --- 4. COMPONENT BREAKDOWN (SECTION 3) ---
    ui.hr(),
    ui.h3("3. Component Breakdown: Identifying Specific Vulnerability Drivers"),
    ui.markdown("""
    This view is critical for **anti-erasure practice**, showing exactly which component 
    (Economic, Geographic, Vehicle, Transit, Internet, Roads) drives the score for each tract.
    """),
    
    ui.input_select("chart_geo", "Select a Tract GEOID:", choices=GEOID_CHOICES, selected=list(GEOID_CHOICES.keys())[0] if GEOID_CHOICES else None),
    
    # Grid: Chart on Left, Metrics on Right (or below)
    ui.layout_columns(
        ui.card(
            output_widget("chart_breakdown"),
        ),
        ui.layout_columns(
            ui.value_box(
                "Weighted Score",
                ui.output_text("val_weighted"),
                theme="primary"
            ),
            ui.value_box(
                "Unweighted Score",
                ui.output_text("val_unweighted"),
                theme="secondary"
            ),
            ui.value_box(
                "Poverty Rate",
                ui.output_text("val_poverty"),
                theme="info"
            ),
            col_widths=[12, 12, 12] # Stack vertically next to chart
        ),
        col_widths=[8, 4]
    ),

    # --- 5. DATA TABLE (SECTION 4) ---
    ui.hr(),
    ui.h3("4. Full Data Table"),
    ui.card(
        ui.output_data_frame("stat_table")
    ),

    # --- 6. METHODOLOGY (SECTION 5) ---
    ui.hr(),
    ui.h3("5. Model Methodology"),
    ui.markdown(r"""
    All standardized variables use global mean (\(\mu\)) and global standard deviation (\(\sigma\)) from the master dataset.  
    This ensures Decatur County's tracts (N=8) are evaluated relative to the broader landscape rather than an artificially small local sample.

    #### Model Formulas
    **Final Weighted Model**
    $$V_{final\_W} = C_{Eco} + C_{Geo} + TVS$$

    **Transport Vulnerability Score**
    $$TVS = 0.5\,C_A + 0.2\,C_B + 0.2\,C_{Internet\_C} + 0.1\,C_{Roads\_D}$$

    **Transit Absence Penalty**
    $$C_{Transit\_B} = +3.0$$

    **Unweighted Model (for comparison)**
    $$V_{final\_UW} = C_{Eco} + C_{Geo} + C_A + C_{Transit\_B} + C_{Internet\_C} + C_{Roads\_D}$$
    """),
    
    ui.hr(),
    ui.h4("Component Definitions"),
    ui.markdown(r"""
    **1. Economic Vulnerability (C_Economic)**
    $$C_{Economic} = -Z_{Income} + Z_{Poverty\_Pct}$$
    *Concept:* Income alone hides deprivation. Pairing reversed income with positive poverty rate prevents flattening of economic reality.

    **2. Geographic Vulnerability (C_Geo)**
    $$C_{Geo} = -Z_{den\_totalfoodstores}$$
    *Concept:* Store density directly indicates food resource availability. Low density increases vulnerability.

    **3. Vehicle Access Vulnerability (C_Vehicle_A)**
    $$C_{Vehicle\_A} = Z_{Vehicles\_PctHH\_NoVehicle}$$
    *Concept:* In rural areas without transit, lacking a vehicle is one of the strongest predictors of limited access.

    **4. Transit Access Vulnerability (C_Transit_B)**
    $$C_{Transit\_B} = -Z_{STOPS\_PER\_CAPITA}$$
    *(Penalty +3.0 when no transit)*
    *Concept:* Transit stop availability is protective. Zero availability indicates systemic infrastructure failure.

    **5. Internet Access Vulnerability (C_Internet_C)**
    $$C_{Internet\_C} = Z_{est\_NO\_INT} + Z_{pct\_no\_internet} - Z_{pct\_cellular\_broadband}$$
    *Concept:* Captures multiple facets of digital exclusion. Accessing transit can be entirely data access dependent. Digital connectivity acknowledges this contextual reality. Lack-of-access increases lack of mobility potential.

    **6. Roads / Walkability Vulnerability (C_Roads_D)**
    $$C_{Roads\_D} = -Z_{PROP\_PRIM\_SEC\_ROADS}$$
    *Concept:* Primary and secondary roads serve as a proxy for mobility infrastructure. Low road density increases vulnerability.
    """),

    ui.hr(),
    ui.h4("Interpretation Framework"),
    ui.markdown("""
    The comparison between **unweighted** and **RUCA-weighted** final scores reveals the necessity of context-sensitive weighting.  
    The unweighted model treats all components equally, erasing rural conditions.  
    The weighted model elevates the components that matter most in Micropolitan regions—vehicle access, roads—and down-weights structurally absent elements like transit.

    This prevents the erasure of localized transport barriers and creates an equitable, context-aware food access vulnerability score.
    """),
    
    ui.hr(),
    ui.h3("6. Datasets"),
    ui.markdown("""
    The following external datasets were used to construct the vulnerability index:

    | Dataset | Contribution | Source |
    |:---|:---|:---|
    | **ACS 5-Year Estimates (2019-2023)** | Socioeconomic indicators | [U.S. Census Bureau](https://data.census.gov) |
    | **NaNDA Grocery Stores (2020)** | Grocery store density | [ICPSR](https://doi.org/10.3886/E209313V1) |
    | **NaNDA Transit Stops (2024)** | Public transit stop counts | [ICPSR](https://doi.org/10.3886/ICPSR38605.v2) |
    | **NaNDA Road Infrastructure (2020)** | Road density (walkability proxy) | [ICPSR](https://doi.org/10.3886/ICPSR38585.v2) |
    | **NaNDA Internet Access (2019)** | Internet connectivity | [ICPSR](https://doi.org/10.3886/ICPSR38559.v1) |
    | **USDA RUCA Codes (2020)** | Rural-urban connectivity | [USDA ERS](https://www.ers.usda.gov/data-products/rural-urban-commuting-area-codes/) |
    | **USDA RUCC Codes (2023)** | County-level classification | [USDA ERS](https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/) |
    | **HUD ZIP-Tract Crosswalk (2020)** | Geographic alignment | [HUD User](https://www.huduser.gov/portal/datasets/usps_crosswalk.html) |
    """),
    
    # --- FOOTER ---
    ui.hr(),
    ui.div(
        ui.HTML("<p style='text-align:center'><a href='https://github.com/phinnphace/MVP-Food-access-county' target='_blank'>GitHub Repository</a></p>"),
        style="padding-bottom: 20px;"
    )
)

# ==============================================================================
# SERVER
# ==============================================================================

def server(input, output, session):
    
    # 1. Update Chart Selector (Use TRACTCE20 for Dictionary Labels)
    @reactive.effect
    def _():
        if not full_data.empty:
            # Create dict: {GEOID: "Tract 960100"}
            choices = {
                row['GEOID']: f"Tract {row.get('TRACTCE20', 'Unknown')}" 
                for _, row in full_data.iterrows()
            }
            # Sort by label
            sorted_geoids = sorted(choices.keys(), key=lambda k: choices[k])
            ui.update_select("chart_geo", choices=choices, selected=sorted_geoids[0] if sorted_geoids else None)
            ui.update_select("geo_compare", choices=choices, selected=sorted_geoids[0] if sorted_geoids else None)
    
    # --- CHART 1: COMPARISON (Static) ---
    @render_altair
    def chart_comp():
        if full_data.empty: return None
        
        # Prepare Data: Melting Weighted and Unweighted
        # Check column names carefully: 'V_final_Weighted' vs 'Final_Weighted_Score'
        # Based on check, use 'V_final_Weighted' if available, else 'Final_Weighted_Score'
        col_w = 'V_final_Weighted' if 'V_final_Weighted' in full_data.columns else 'Final_Weighted_Score'
        col_u = 'V_final_Unweighted' if 'V_final_Unweighted' in full_data.columns else 'Final_Unweighted_Score'
        
        comp_data = full_data[['GEOID', col_u, col_w]].copy()
        # Rename for cleaner legend if needed, but melting handles it
        comp_melt = comp_data.melt(id_vars='GEOID', var_name='Score Type', value_name='Score')
        
        # Clean labels
        comp_melt['Score Type'] = comp_melt['Score Type'].replace({
            col_w: 'Weighted Score',
            col_u: 'Unweighted Score'
        })

        c = alt.Chart(comp_melt).mark_bar().encode(
            x=alt.X('GEOID:N', sort=alt.EncodingSortField(field="Score", op="max", order="descending"), title='Tract GEOID'),
            y=alt.Y('Score:Q', title='Vulnerability Score'),
            color=alt.Color('Score Type:N', scale=alt.Scale(range=['#007ACC', '#D34D4D']), legend=alt.Legend(title="Score Type")),
            xOffset='Score Type:N', # Grouped bar
            tooltip=['GEOID', 'Score Type', alt.Tooltip('Score', format=".2f")]
        ).properties(
            title="Unweighted vs. Weighted (TVS) Vulnerability Scores",
            height=400
        ).interactive()
        
        return c

    # --- CHART 2: COMPONENT BREAKDOWN (Dynamic) ---
    @render_altair
    def chart_breakdown():
        geo_id = input.chart_geo()
        if not geo_id or full_data.empty: return None
        
        row = full_data[full_data['GEOID'] == geo_id]
        if row.empty: return None
        row = row.iloc[0]
        
        comp_list = ['C_Economic', 'C_Geographic', 'C_Vehicle_A', 'C_Transit_B', 'C_Internet_C', 'C_Roads_D']
        available_comps = [c for c in comp_list if c in full_data.columns]
        
        if not available_comps: return None
            
        chart_data = pd.DataFrame({
            'Component': available_comps, 
            'Vulnerability Score': [row.get(c, 0) for c in available_comps]
        })
        
        c = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Vulnerability Score:Q', title='Z-Score'),
            y=alt.Y('Component:N', sort=None, title=''),
            color=alt.condition(
                alt.datum['Vulnerability Score'] > 0,
                alt.value('#D34D4D'),
                alt.value('#2E8B57')
            ),
            tooltip=['Component', alt.Tooltip('Vulnerability Score', format=".2f")]
        ).properties(
            title=f"Component Scores for Tract",
            height=300
        )
        return c

    # --- METRICS (Dynamic) ---
    def get_selected_row():
        geo_id = input.chart_geo()
        if not geo_id or full_data.empty: return None
        row = full_data[full_data['GEOID'] == geo_id]
        if row.empty: return None
        return row.iloc[0]

    @render.text
    def val_weighted():
        row = get_selected_row()
        if row is None: return "N/A"
        col = 'V_final_Weighted' if 'V_final_Weighted' in row else 'Final_Weighted_Score'
        return f"{row.get(col, 0):.2f}"

    @render.text
    def val_unweighted():
        row = get_selected_row()
        if row is None: return "N/A"
        col = 'V_final_Unweighted' if 'V_final_Unweighted' in row else 'Final_Unweighted_Score'
        row = get_selected_row()
        if row is None: return "N/A"
        col = 'V_final_Unweighted' if 'V_final_Unweighted' in row else 'Final_Unweighted_Score'
        return f"{row.get(col, 0):.2f}"

    @render.ui
    def lila_callout():
        # Default to geo_compare input, but could be redundant
        geo_id = input.geo_compare()
        if not geo_id or full_data.empty: 
            return ui.h3("Select a Tract", style="color: #666;")
        
        row = full_data[full_data['GEOID'] == geo_id]
        if row.empty: return None
        row = row.iloc[0]
        
        is_lila = row.get('LILA_Flag', 0)
        
        if is_lila == 1:
            return ui.div(
                ui.h3("YES: USDA Food Desert", style="color: #d62728; font-weight: bold; margin: 0;"),
                ui.p("Low Income + Low Access (> 10mi)", style="margin: 0; color: #666; font-size: 12px;")
            )
        else:
             return ui.div(
                ui.h3("NO: Not a Food Desert", style="color: #2ca02c; font-weight: bold; margin: 0;"),
                ui.p("Does not meet Low Income & Low Access criteria", style="margin: 0; color: #666; font-size: 12px;")
            )

    @render.text
    def val_poverty():
        row = get_selected_row()
        if row is None: return "N/A"
        return f"{row.get('poverty_pct', 0):.1f}%"

    # --- DATA TABLE ---
    @render.data_frame
    def stat_table():
        if full_data.empty: return render.DataGrid(pd.DataFrame({"Error": ["No Data"]}))
        
        # Show ALL columns except geometry, prioritizing TRACTCE20
        df_show = full_data.drop(columns=['geometry'], errors='ignore').copy()
        
        if 'TRACTCE20' in df_show.columns:
            cols = ['TRACTCE20'] + [c for c in df_show.columns if c != 'TRACTCE20']
            df_show = df_show[cols]
        
        numeric_cols = df_show.select_dtypes(include=[np.number]).columns
        df_show[numeric_cols] = df_show[numeric_cols].round(3) # Match Streamlit rounding
        
        return render.DataGrid(df_show, selection_mode="none")
    
    # ... [Keep existing Map Logic Below] ...
    
    # Selected Variable State (Start Empty)
    selected_var = reactive.value(None)

    # --- Force Clear on Startup ---
    # This effect runs once when the session starts to ensure
    # radio buttons are actually cleared (fixing the "Always On" issue).
    @reactive.effect
    def _():
        ui.update_radio_buttons("rb_assess", selected="_none_")
        ui.update_radio_buttons("rb_comp", selected="_none_")
        ui.update_radio_buttons("rb_systemic", selected="_none_")
    
    # --- Clear Button ---
    @reactive.effect
    @reactive.event(input.btn_clear)
    def _():
        # Reset all radio buttons
        ui.update_radio_buttons("rb_assess", selected="_none_")
        ui.update_radio_buttons("rb_comp", selected="_none_")
        ui.update_radio_buttons("rb_systemic", selected="_none_")
        # Reset state
        selected_var.set(None)

    # --- Radio Button Logic (Mutually Exclusive) ---
    @reactive.effect
    @reactive.event(input.rb_assess)
    def _():
        if input.rb_assess() and input.rb_assess() != "_none_":
            ui.update_radio_buttons("rb_comp", selected="_none_")
            ui.update_radio_buttons("rb_systemic", selected="_none_")
            selected_var.set(input.rb_assess())

    @reactive.effect
    @reactive.event(input.rb_comp)
    def _():
        if input.rb_comp() and input.rb_comp() != "_none_":
            ui.update_radio_buttons("rb_assess", selected="_none_")
            ui.update_radio_buttons("rb_systemic", selected="_none_")
            selected_var.set(input.rb_comp())

    @reactive.effect
    @reactive.event(input.rb_systemic)
    def _():
        if input.rb_systemic() and input.rb_systemic() != "_none_":
            ui.update_radio_buttons("rb_assess", selected="_none_")
            ui.update_radio_buttons("rb_comp", selected="_none_")
            selected_var.set(input.rb_systemic())

    # --- Map Renderer ---
    @render.ui
    def main_map_ui():
        try:
            var_id = selected_var()
            
            if full_data.empty:
                return ui.HTML("<div style='color:red;'>Error: Could not load final_app_data.geojson</div>")

            # --- GENERATE MAP (Base) ---
            m = folium.Map(location=[30.87, -84.57], zoom_start=10, tiles="OpenStreetMap")

            # CSS Hack for Labels
            m.get_root().header.add_child(folium.Element("""
                <style>
                    .static-label {
                        font-size: 12px !important;
                        font-weight: bold !important;
                        color: black !important;
                        text-shadow: 2px 2px 0px white;
                        white-space: nowrap;
                        text-align: center;
                        pointer-events: none;
                    }
                </style>
            """))

            # CASE A: Blank Map (No Variable Selected)
            if var_id is None or var_id == "_none_":
                # Just draw outlines
                folium.GeoJson(
                    full_data,
                    style_function=lambda x: {
                        'fillColor': 'transparent',
                        'color': 'black',
                        'weight': 2,
                        'fillOpacity': 0
                    },
                    tooltip=folium.GeoJsonTooltip(fields=['TRACTCE20'], aliases=['Tract:'])
                ).add_to(m)

            # CASE B: Choropleth (Variable Selected)
            else:
                # Find Config
                config = None
                for group in VAR_CONFIG.values():
                    for item in group:
                        if item["id"] == var_id:
                            config = item
                            break
                    if config: break
                
                # Render Choropleth
                if config and var_id in full_data.columns:
                    colormap = create_jenks_colormap(full_data[var_id], var_id, caption=config["label"])
                    m.add_child(colormap)
                    
                    folium.GeoJson(
                        full_data,
                        style_function=lambda feature: {
                            'fillColor': colormap(feature['properties'].get(var_id, 0)),
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0.7
                        },
                        # REMOVED SCORE FROM TOOLTIP
                        tooltip=folium.GeoJsonTooltip(fields=['TRACTCE20'], aliases=['Tract:'])
                    ).add_to(m)

            # Optional: Grocery Points (Always Show)
            if grocery_gdf is not None:
                for _, row in grocery_gdf.iterrows():
                    if row.geometry:
                        folium.CircleMarker(
                            location=[row.geometry.y, row.geometry.x],
                            radius=4,
                            color='black',
                            fill=True,
                            fill_color='green',
                            fill_opacity=1,
                            tooltip="Grocery Store"
                        ).add_to(m)

            return ui.HTML(m._repr_html_())

        except Exception as e:
            return ui.HTML(f"<div style='color:red;'><h3>Map Error</h3>{e}</div>")

app = App(app_ui, server)
