import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

plt.rcParams["font.family"] = "DejaVu Sans"

def most_common(series):
    """Returns the most frequent value (mode) from a pandas Series."""
    mode_vals = series.mode()
    return mode_vals.iloc[0] if len(mode_vals) > 0 else None

# Load data
lfs_file = "resources/LFS-PUF-December-2023.csv"
income_expenditure_file = "resources/Family-Income-and-Expenditure.csv"

df_lfs = pd.read_csv(lfs_file)
df_income_expenditure = pd.read_csv(income_expenditure_file)

# Ensure correct column names
income_column = "Total Household Income"
food_expenditure_column = "Total Food Expenditure"

# Estimate total non-food expenditure
non_food_expenditure_columns = [
    "Housing and water Expenditure",
    "Medical Care Expenditure",
    "Education Expenditure",
    "Transportation Expenditure",
    "Miscellaneous Goods and Services Expenditure"
]
df_income_expenditure["Total Non-Food Expenditure"] = df_income_expenditure[non_food_expenditure_columns].sum(axis=1)

# Define income bins and labels
income_bins = [0, 50000, 150000, 300000, 500000, 1000000, float("inf")]
income_labels = ["<50K", "50K-150K", "150K-300K", "300K-500K", "500K-1M", "1M+"]
df_income_expenditure["Income Range"] = pd.cut(df_income_expenditure[income_column], bins=income_bins, labels=income_labels)

# Define Income Groups for interactive part
def categorize_income(income):
    if income < 20000:
        return 'Low Income'
    elif 20000 <= income < 60000:
        return 'Lower Middle Income'
    elif 60000 <= income < 150000:
        return 'Upper Middle Income'
    else:
        return 'High Income'

df_income_expenditure['Income Group'] = df_income_expenditure['Total Household Income'].apply(categorize_income)

# Region name corrections
region_corrections = {
    "IVB - MIMAROPA": "IV-B - MIMAROPA",
    "IVA - CALABARZON": "IV-A - CALABARZON",
    "IX - Zasmboanga Peninsula": "IX - Zamboanga Peninsula",
    " ARMM": "ARMM",
    "Caraga": "Region XIII  (Caraga)"
}
df_income_expenditure["Region"] = df_income_expenditure["Region"].replace(region_corrections)

# Group data
df_income_expenditure_grouped = df_income_expenditure.groupby("Region").agg(
    {"Total Household Income": "mean", "Total Food Expenditure": "mean"}
).reset_index()
df_income_expenditure_grouped.columns = ["Region", "Average Income", "Average Expenditure"]

# Clean and compute unemployment rate
df_lfs_clean = df_lfs.dropna(subset=["PUFNEWEMPSTAT"])
df_lfs_clean["PUFNEWEMPSTAT"] = df_lfs_clean["PUFNEWEMPSTAT"].astype(str).str.strip()
df_lfs_clean = df_lfs_clean[df_lfs_clean["PUFNEWEMPSTAT"].str.isnumeric()]
df_lfs_clean["PUFNEWEMPSTAT"] = df_lfs_clean["PUFNEWEMPSTAT"].astype(int)

labor_force = df_lfs_clean.groupby("PUFREG")["PUFNEWEMPSTAT"].count()
unemployed = df_lfs_clean[df_lfs_clean["PUFNEWEMPSTAT"].isin([2, 3])].groupby("PUFREG")["PUFNEWEMPSTAT"].count()
unemployed = unemployed.reindex(labor_force.index, fill_value=0)
unemployment_rate = (unemployed / labor_force) * 100

df_unemployment = unemployment_rate.reset_index()
df_unemployment.columns = ["Region_Code", "Unemployment Rate"]

region_mapping = {
    13: "NCR", 14: "CAR", 1: "I - Ilocos Region", 2: "II - Cagayan Valley",
    3: "III - Central Luzon", 4: "IV-A - CALABARZON", 17: "IV-B - MIMAROPA",
    5: "V - Bicol Region", 6: "VI - Western Visayas", 7: "VII - Central Visayas",
    8: "VIII - Eastern Visayas", 9: "IX - Zamboanga Peninsula", 10: "X - Northern Mindanao",
    11: "XI - Davao Region", 12: "XII - SOCCSKSARGEN", 16: "Region XIII  (Caraga)", 19: "ARMM"
}
df_unemployment["Region"] = df_unemployment["Region_Code"].map(region_mapping)
df_unemployment = df_unemployment.drop(columns=["Region_Code"], errors="ignore")

# Merge datasets
df_final_combined = pd.merge(df_income_expenditure_grouped, df_unemployment, on="Region", how="outer")
df_final_cleaned = df_final_combined[df_final_combined["Unemployment Rate"].notna()]
df_final_cleaned = df_final_cleaned.sort_values(by="Average Income", ascending=False)

# Group data with new columns
df_income_expenditure_grouped = (
    df_income_expenditure
    .groupby("Region")
    .agg({
        "Total Household Income": "mean",
        "Total Food Expenditure": "mean",
        "Household Head Highest Grade Completed": most_common
    })
    .reset_index()
)

# Rename the aggregated columns
df_income_expenditure_grouped.columns = [
    "Region",
    "Mean Household Income",
    "Mean Household Expenditure",
    "Most Common HH Head Education"
]

# Custom Color Palette
color_palette = ['#1b4965', '#022b3a', '#1f7a8c', '#5fa8d3', '#bee9e8', '#52b69a', '#34a0a4', '#b5e48c', '#70a288']

# === STYLING CONFIG ===
background_color = "#fff9ed"
income_color = "#1b4965"
expenditure_color = "#5fa8d3"
unemployment_line_color = "#52b69a"
unemployment_marker_face = "#1f7a8c"
unemployment_marker_edge = "#022b3a"


# Group data with new columns
df_income_expenditure_grouped = (
    df_income_expenditure
    .groupby("Region")
    .agg({
        "Total Household Income": "mean",
        "Total Food Expenditure": "mean",
        "Household Head Highest Grade Completed": most_common
    })
    .reset_index()
)
# Rename the aggregated columns
df_income_expenditure_grouped.columns = [
    "Region",
    "Mean Household Income",
    "Mean Household Expenditure",
    "Most Common HH Head Education"
]

df_final_combined = pd.merge(
    df_income_expenditure_grouped,
    df_unemployment,
    on="Region",
    how="outer"
)
df_final_cleaned_chloropeth = df_final_combined[df_final_combined["Unemployment Rate"].notna()]
df_final_cleaned_chloropeth = df_final_cleaned_chloropeth.sort_values(by="Mean Household Income", ascending=False)

# import gejson PH regions
geojson_path = "resources/ph_regions.json"
# Read the GeoJSON file
import json
with open(geojson_path, 'r') as f:
    geojson = json.load(f)


name_list = []

for feature in geojson['features']:
    name_list.append(feature['properties']['REGION'])

# Renames the regions based on what is being used in
region_mapping = {
    'NCR': 'Metropolitan Manila',
    'IV-A - CALABARZON': 'CALABARZON (Region IV-A)',
    'III - Central Luzon': 'Central Luzon (Region III)',
    'CAR': 'Cordillera Administrative Region (CAR)',
    'XI - Davao Region': 'Davao Region (Region XI)',
    'I - Ilocos Region': 'Ilocos Region (Region I)',
    'II - Cagayan Valley': 'Cagayan Valley (Region II)',
    'VII - Central Visayas': 'Central Visayas (Region VII)',
    'VI - Western Visayas': 'Western Visayas (Region VI)',
    'IV-B - MIMAROPA': 'MIMAROPA (Region IV-B)',
    'X - Northern Mindanao': 'Northern Mindanao (Region X)',
    'Region XIII  (Caraga)': 'Caraga (Region XIII)',
    'VIII - Eastern Visayas': 'Eastern Visayas (Region VIII)',
    'IX - Zamboanga Peninsula': 'Zamboanga Peninsula (Region IX)',
    'V - Bicol Region': 'Bicol Region (Region V)',
    'XII - SOCCSKSARGEN': 'SOCCSKSARGEN (Region XII)',
    'ARMM': 'Autonomous Region of Muslim Mindanao (ARMM)'
}

# replace df_final_cleaned['Region'] with the map

df_final_cleaned_chloropeth['Region'] = df_final_cleaned_chloropeth['Region'].map(region_mapping)

df_final_cleaned_chloropeth

biv_bins_map = {
    "A3": "rgba(31,122,140,1)",  # #1f7a8c
    "B3": "rgba(27,73,101,1)",   # #1b4965
    "C3": "rgba(2,43,58,1)",     # #022b3a

    "A2": "rgba(112,162,136,1)", # #70a288
    "B2": "rgba(52,160,164,1)",  # #34a0a4
    "C2": "rgba(82,182,154,1)",  # #52b69a

    "A1": "rgba(190,233,232,1)", # #bee9e8
    "B1": "rgba(181,228,140,1)", # #b5e48c
    "C1": "rgba(95,168,211,1)",  # #5fa8d3

    "ZZ": "rgba(253,240,213,1)"  # #fdf0d5
}

def create_legend(fig, colors):

    #Vertical position of top right corner (0: bottom, 1: top)
    top_rt_vt = 0.95
    #Horizontal position of top right corner (0: left, 1: right)
    top_rt_hz = 1.0

    #reverse the order of colors
    legend_colors = colors[:]
    legend_colors.reverse()

    #calculate coordinates for all nine rectangles
    coord = []

    #adapt height to ratio to get squares
    width = 0.04
    height = 0.04/0.8

    #start looping through rows and columns to calculate corners the squares
    for row in range(1, 4):
        for col in range(1, 4):
            coord.append({
                'x0': round(top_rt_vt-(col-1)*width, 4),
                'y0': round(top_rt_hz-(row-1)*height, 4),
                'x1': round(top_rt_vt-col*width, 4),
                'y1': round(top_rt_hz-row*height, 4)
            })

    #create shapes (rectangle)
    for i, value in enumerate(coord):

        #add rectangle
        fig.add_shape(go.layout.Shape(
            type='rect',
            fillcolor=legend_colors[i],
            line=dict(
                color='#f8f8f8',
                width=0,
            ),
            xref = 'paper',
            yref = 'paper',
            xanchor = 'right',
            yanchor = 'top',
            x0 = coord[i]['x0'],
            y0 = coord[i]['y0'],
            x1 = coord[i]['x1'],
            y1 = coord[i]['y1'],
        ))

        #add text for first variable
        fig.add_annotation(
            xref='paper',
            yref='paper',
            xanchor='left',
            yanchor='top',
            x=coord[8]['x1'],
            y=coord[8]['y1'],
            showarrow=False,
            text="Household Income"  + ' →',
            font=dict(
                color='#000',
                size=12
            ),
            borderpad=1,
        )

        #add text for second variable
        fig.add_annotation(
            xref='paper',
            yref='paper',
            xanchor='right',
            yanchor='bottom',
            x=coord[8]['x1'],
            y=coord[8]['y1'],
            showarrow=False,
            text="Education Level" + ' →',
            font=dict(
                color='#000',
                size=12,
            ),
            textangle=270,
            borderpad=1
        )
    return fig

def generate_bivariate_map(gdf, biv_bins_col, color_discrete, colors_scheme,
              custom_data_hover, map_title, map_subtitle, geojson):
    """
    Function to create map
    Arguments:
        gdf (GeoPandas DataFrame): Geospatial data, index as location and geometry col with polygon data
        biv_bins_col (list: str): color scheme to use in the bivariate map, list length of 9
        color_discrete (list: str): Dictionary mapping bivariate bin values to colors.
        colors_scheme (list) : color scheme to use in bivariate map
        custom_data_hover (list: str): data to be used in hover, ex. ["Zipcode", "Client_Count", "Age", "VL"]
        map_title (string): title for map
        map_subtitle (string): subtitle for map
    Returns:
        Plotly Figure Object
    """
    fig = px.choropleth(
        gdf,
        geojson=geojson,
        locations='Region',
        featureidkey='properties.REGION',
        color=biv_bins_col,
        height=885, width=1000,
        color_discrete_map = color_discrete,
        hover_data = custom_data_hover,
    ).update_layout(
        geo=dict(
        fitbounds="locations",
        visible=False # make the base map invisible so it looks cleaner
        ),
        showlegend=False,
        title_x=0.05,
        title=dict(
            text=map_title,
            font=dict(
                size=24
            ),
        ),
        title_subtitle=dict(
            text=map_subtitle,
            font=dict(size=16)
        ),
        margin={"r":0, "t":85, "l":0, "b":0},
        map_style="carto-darkmatter",
        autosize=False,
        newshape_line_color="yellow",
        modebar_add = ["drawline", "drawopenpath", "drawclosedpath", "drawcircle", "drawrect", "eraseshape"],
        modebar={"orientation":"h", "bgcolor":"white", "color":"black", "activecolor":"#9ed3cd"}
    ).update_traces(
        marker_line_width=0.5, # width of the geo entity borders
        marker_line_color="#d1d1d1", # color of the geo entity borders
        showscale=False, #hide the colorscale
    )

    #add legend
    fig = create_legend(fig, colors_scheme)

    return fig

# function to get bivariate color given two percentiles
percentile_bounds1 = [0.33, 0.66, 1] # variable 1
percentile_bounds2 = [0.33, 0.66, 1] # variable 2

def get_bivariate_choropleth_color_tester(p1, p2):
    if p1>=0 and p2>=0:
        count = 0
        stop = False
        for percentile_bound_p1 in percentile_bounds1:
            for percentile_bound_p2 in percentile_bounds2:
                if (not stop) and (p1 <= percentile_bound_p1):
                    if (not stop) and (p2 <= percentile_bound_p2):
                        color = count
                        stop = True
                count += 1
    else:
        color = -1
    return color

# dividing Mean household income to 3 bins, low household, medium household, high household

df_final_cleaned_chloropeth['Binned Household Income'] = pd.qcut(
    df_final_cleaned_chloropeth['Mean Household Income'],
    q=3,  # 3 quantiles => 3 bins
    labels=['Low', 'Medium', 'High']
)


# Define numeric encodings for each category
education_map = {
    "Elementary Graduate": 0.25,
    "High School Graduate": 0.50,
    "College Graduate": 0.75
}

income_map = {
    "Low": 0.25,
    "Medium": 0.50,
    "High": 0.75
}

# Create numeric columns in your DataFrame
df_final_cleaned_chloropeth["EduVal"] = df_final_cleaned_chloropeth["Most Common HH Head Education"].map(education_map)
df_final_cleaned_chloropeth["IncVal"] = df_final_cleaned_chloropeth["Binned Household Income"].map(income_map)

# Use your existing function to get an integer code (0..8) for each row
df_final_cleaned_chloropeth["Bivariate Numeric Code"] = df_final_cleaned_chloropeth.apply(
    lambda row: get_bivariate_choropleth_color_tester(row["EduVal"], row["IncVal"]),
    axis=1
)

# Convert the integer code to the bivariate keys (e.g. "A1", "A2", ... "C3")
#    The order depends on how your function increments 'count'.
#    For example, if 'count' increments row-wise:
#    0 -> A1, 1 -> A2, 2 -> A3, 3 -> B1, 4 -> B2, etc.
numeric_to_bin = {
    0: "A1", 1: "A2", 2: "A3",
    3: "B1", 4: "B2", 5: "B3",
    6: "C1", 7: "C2", 8: "C3",
    -1: "ZZ"  # If your function returns -1 for invalid
}

df_final_cleaned_chloropeth["Bivariate Bin"] = df_final_cleaned_chloropeth["Bivariate Numeric Code"].map(numeric_to_bin)

# Finally, map the bivariate bin to your color dictionary
df_final_cleaned_chloropeth["Bivariate Color"] = df_final_cleaned_chloropeth["Bivariate Bin"].map(biv_bins_map)

# Now df_final_cleaned["Bivariate Color"] has the RGBA color codes
# you can use for your choropleth shading.

map_title = 'Regional Relationship Between Education and Income in the Philippines'
map_subtitle = 'Mode Education Level vs. Average Household Income'

biv_bins_map = {
    "A1": "#e2f2e3",  # (Light Green + Light Blue)
    "A2": "#c1e0da",
    "A3": "#9ec9dd",
    "B1": "#bad3af",
    "B2": "#a9d1b8",
    "B3": "#69adaf",
    "C1": "#88c685",
    "C2": "#6fb998",
    "C3": "#4c9e8b",
    "ZZ": "#fdf0d5"   # No-data or fallback
}


colors = [
    "#e2f2e3",  # A1 (lowest Edu, lowest Income)
    "#c1e0da",  # A2 (lowest Edu, medium Income)
    "#9ec9dd",  # A3 (lowest Edu, highest Income)
    "#bad3af",  # B1 (medium Edu, lowest Income)
    "#a9d1b8",  # B2 (medium Edu, medium Income)
    "#69adaf",  # B3 (medium Edu, highest Income)
    "#88c685",  # C1 (highest Edu, lowest Income)
    "#6fb998",  # C2 (highest Edu, medium Income)
    "#4c9e8b"   # C3 (highest Edu, highest Income)
]

hover_data_dict = ["Region", "Mean Household Income", "Most Common HH Head Education"]


fig = generate_bivariate_map(
    gdf = df_final_cleaned_chloropeth,
    biv_bins_col = 'Bivariate Bin',
    color_discrete = biv_bins_map,
    colors_scheme=colors,
    custom_data_hover=hover_data_dict,
    map_title=map_title,
    map_subtitle=map_subtitle,
    geojson=geojson,
)


# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Income, Expenditure & Labor Force Dashboard"

# Prepare data for violin chart
income_counts = df_income_expenditure["Income Range"].value_counts().reindex(income_labels)

# slider fixed range for unemployment
slider_min = 0
slider_max = 45  # Changed to include more range
slider_step = 5
slider_marks = {i: f"{i}%" for i in range(slider_min, slider_max + 1, slider_step)}
all_unique_regions = sorted(df_final_cleaned["Region"].unique())

# Create the combined app layout
app.layout = html.Div(style={'backgroundColor': '#fff9ed', 'fontFamily': 'DejaVu Sans', 'padding': '20px'}, children=[
    html.H1("Philippine Household Income, Expenditure, and Labor Force Dashboard", 
            style={'textAlign': 'center', 'color': '#022b3a', 'marginBottom': '20px'}),
    
    # Tabs for different analysis views
    dcc.Tabs([
        # TAB 1: INCOME DISTRIBUTION ANALYSIS
        dcc.Tab(label="Income Distribution Analysis", children=[
            html.Div([
                html.H2("Household Income Distribution Analysis", style={'textAlign': 'center', 'color': '#022b3a', 'marginTop': '20px'}),
                
                # CONTROLS
                html.Div([
                    html.Label("Select Region:", style={'color': '#1b4965', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='region-filter-tab1',
                        options=[{'label': r, 'value': r} for r in df_income_expenditure['Region'].unique()],
                        multi=True,
                        placeholder="Select regions...",
                        style={'backgroundColor': '#bee9e8', 'color': '#022b3a'}
                    ),
                ], style={'width': '50%', 'margin': '10px auto'}),
                
                # CHARTS
                html.Div([
                    dcc.Graph(id='income-bar-chart'),
                    dcc.Graph(id='expenditure-violin'),
                ]),
                
                # Bar Chart that was clickable in the original
                html.H3("Income Groups Distribution", style={'textAlign': 'center', 'color': '#022b3a', 'marginTop': '30px'}),
                dcc.Graph(
                    id="bar-chart",
                    figure=go.Figure(
                        data=[
                            go.Bar(
                                x=income_labels,
                                y=income_counts.values,
                                marker=dict(
                                    color=["#1b4965", "#1f7a8c", "#5fa8d3", "#34a0a4", "#b5e48c", "#ef476f"],
                                    line=dict(color='black', width=1),
                                ),
                                hoverinfo='x+y'
                            )
                        ],
                        layout=go.Layout(
                            title="Number of Households by Income Group",
                            xaxis_title="Income Range (PHP)",
                            yaxis_title="Number of Households",
                            plot_bgcolor='#fff9ed',
                            paper_bgcolor='#fff9ed',
                            font=dict(size=12)
                        )
                    )
                ),
                
                # Violin Chart that was updated by the bar chart click
                dcc.Graph(id="violin-chart")
            ])
        ]),
        
        # TAB 2: REGIONAL ANALYSIS
        dcc.Tab(label="Regional Analysis", children=[
            html.Div([
                html.H2("Regional Income, Expenditure, and Unemployment Analysis", 
                        style={'textAlign': 'center', 'color': '#022b3a', 'marginTop': '20px'}),
                
                # CONTROLS
                html.Div([
                    html.Label("Filter by Unemployment Rate (%)", style={"fontWeight": "bold"}),
                    dcc.RangeSlider(
                        id='unemployment-slider',
                        min=slider_min,
                        max=slider_max,
                        step=slider_step,
                        value=[slider_min, slider_max],
                        marks=slider_marks
                    ),
                ], style={'width': '80%', 'margin': 'auto', 'padding': '20px'}),
                
                html.Div([
                    html.Label("Regions:", style={"fontWeight": "bold"}),
                    dcc.Checklist(
                        id="region-checklist",
                        options=[{"label": r, "value": r} for r in all_unique_regions],
                        value=all_unique_regions,
                        labelStyle={"display": "block"}
                    )
                ], style={'width': '70%', 'margin': 'auto', 'padding': '20px'}),
                
                # CHARTS
                dcc.Graph(
                    id='income-expenditure-chart',
                    style={'width': '100%', 'height': '600px', 'margin': 'auto'}
                ),
                dcc.Graph(
                    id='unemployment-chart',
                    style={'width': '100%', 'height': '600px', 'margin': 'auto'}
                ),
                
                # INFO DISPLAY
                html.Div(
                    id='info-display',
                    style={
                        'width': '80%',
                        'margin': '30px auto 50px auto',
                        'padding': '10px',
                        'border': '1px solid #ccc',
                        'borderRadius': '5px',
                        'backgroundColor': '#fff9ed',
                    }
                ),
            ])
        ]),
        # TAB 3: BIVARIATE MAP
        dcc.Tab(label="Bivariate Map", children=[
            html.Div([
                html.H3("Bivariate Choropleth Map", style={'textAlign': 'center', 'color': '#022b3a', 'marginTop': '30px'}),
                dcc.Graph(id="bivariate-map", figure=fig)  # <- Adding the map figure here
            ])
        ]),
    ]),
])

# Callbacks from the first app: Income Distribution Analysis
@app.callback(
    [Output('income-bar-chart', 'figure'),
     Output('expenditure-violin', 'figure')],
    [Input('region-filter-tab1', 'value')]
)
def update_charts_tab1(selected_regions):
    filtered_df = df_income_expenditure if not selected_regions else df_income_expenditure[df_income_expenditure['Region'].isin(selected_regions)]

    # Income Distribution Bar Chart
    income_fig = px.histogram(
        filtered_df, x='Income Group',
        title="Income Distribution Across Households",
        color='Income Group',
        color_discrete_sequence=color_palette,
        barmode='group',
        template='plotly_white'
    )
    income_fig.update_layout(
        font=dict(family="DejaVu Sans", size=14, color="#022b3a"),
        paper_bgcolor="#fff9ed",
        plot_bgcolor="#fff9ed"
    )

    # Expenditure Violin Plot
    expenditure_fig = px.violin(
        filtered_df, y='Total Food Expenditure', x='Income Group',
        title="Expenditure Distribution by Income Group",
        box=True, points="all", color='Income Group',
        color_discrete_sequence=color_palette,
        template='plotly_white'
    )
    expenditure_fig.update_layout(
        font=dict(family="DejaVu Sans", size=14, color="#022b3a"),
        paper_bgcolor="#fff9ed",
        plot_bgcolor="#fff9ed"
    )

    return income_fig, expenditure_fig

# Violin chart update from bar chart click
@app.callback(
    Output("violin-chart", "figure"),
    Input("bar-chart", "clickData")
)
def update_violin(clickData):
    if clickData:
        selected_income_range = clickData["points"][0]["x"]
    else:
        selected_income_range = None  # Show all by default

    violin_fig = go.Figure()
    
    for income_group in income_labels:
        group_data = df_income_expenditure[df_income_expenditure["Income Range"] == income_group][food_expenditure_column]
        stats = group_data.describe()
        color = "#5fa8d3" if income_group != selected_income_range else "#ef476f"
        opacity = 0.5 if income_group != selected_income_range else 0.85
        line_width = 1 if income_group != selected_income_range else 4

        violin_fig.add_trace(go.Violin(
            y=group_data,
            x=[income_group] * len(group_data),
            name=income_group,
            box_visible=True,
            meanline_visible=True,
            line_color=color,
            line_width=line_width,
            fillcolor=color,
            showlegend=False,
            opacity=opacity,
            scalemode="count",
            hovertemplate=(
                f"<b>{income_group}</b><br>"
                f"Median: {stats['50%']:.2f}<br>"
                f"Min: {stats['min']:.2f}<br>"
                f"Max: {stats['max']:.2f}<br>"
                f"IQR: {(stats['75%'] - stats['25%']):.2f}<extra></extra>"
            )
        ))

    violin_fig.update_layout(
        title="Food Expenditure Distribution by Income Group",
        yaxis_title="Total Food Expenditure (PHP)",
        plot_bgcolor='#fff9ed',
        paper_bgcolor='#fff9ed',
        font=dict(size=12)
    )

    return violin_fig

# Callbacks from the second app: Regional Analysis
@app.callback(
    Output("region-checklist", "options"),
    Output("region-checklist", "value"),
    Input("unemployment-slider", "value"),
    State("region-checklist", "value")
)
def update_region_checklist_options(unemp_range, currently_checked):
    low, high = unemp_range
    valid_df = df_final_cleaned[
        (df_final_cleaned["Unemployment Rate"] >= low) &
        (df_final_cleaned["Unemployment Rate"] <= high)
    ]
    allowed_regions = sorted(valid_df["Region"].unique())

    new_options = [{"label": r, "value": r} for r in allowed_regions]
    new_value = [r for r in currently_checked if r in allowed_regions]

    return new_options, new_value

@app.callback(
    Output('income-expenditure-chart', 'figure'),
    Output('unemployment-chart', 'figure'),
    Output('info-display', 'children'),
    Input('unemployment-slider', 'value'),
    Input('region-checklist', 'value'),
    Input('income-expenditure-chart', 'clickData')
)
def update_charts_tab2(unemp_range, selected_regions, click_data):
    low, high = unemp_range

    # Slider filters
    df_slider_filtered = df_final_cleaned[
        (df_final_cleaned["Unemployment Rate"] >= low) &
        (df_final_cleaned["Unemployment Rate"] <= high)
    ]
    # Checklist filters
    filtered_df = df_slider_filtered[df_slider_filtered["Region"].isin(selected_regions)]

    # Grouped bar chart
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=filtered_df["Region"],
        y=filtered_df["Average Income"],
        name="Average Income",
        marker_color="#1b4965",
        hovertemplate="Region: %{x}<br>Income: %{y}<extra></extra>"
    ))
    fig_bar.add_trace(go.Bar(
        x=filtered_df["Region"],
        y=filtered_df["Average Expenditure"],
        name="Average Expenditure",
        marker_color="#5fa8d3",
        hovertemplate="Region: %{x}<br>Expenditure: %{y}<extra></extra>"
    ))
    fig_bar.update_layout(
        barmode='group',
        title="Average Family Income vs. Expenditure",
        xaxis_title="Region",
        yaxis_title="PHP",
        hovermode="x unified",
        font=dict(family="Arial"),
        paper_bgcolor="#fff9ed",
        plot_bgcolor="#fff9ed"
    )
    if len(filtered_df) > 0:
        max_val = max(filtered_df["Average Income"].max(), filtered_df["Average Expenditure"].max())
        fig_bar.update_yaxes(range=[0, max_val * 1.1])
    else:
        fig_bar.update_yaxes(range=[0, 1])

    # Line chart
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=filtered_df["Region"],
        y=filtered_df["Unemployment Rate"],
        mode='lines+markers',
        name="Unemployment Rate",
        marker=dict(size=7, color="#52b69a", line=dict(width=1, color="#022b3a")),
        line=dict(color="#52b69a", width=2),
        hovertemplate="Region: %{x}<br>Unemp Rate: %{y:.2f}%<extra></extra>"
    ))
    fig_line.update_layout(
        title="Unemployment Rate by Region",
        xaxis_title="Region",
        yaxis_title="Unemployment Rate (%)",
        hovermode="closest",
        font=dict(family="Arial"),
        paper_bgcolor="#fff9ed",
        plot_bgcolor="#fff9ed"
    )
    fig_line.update_yaxes(dtick=5)

    # Highlight the selected region
    highlighted_region = None
    if click_data and "points" in click_data:
        highlighted_region = click_data['points'][0]['x']

    if len(filtered_df) > 0 and len(fig_line.data) > 0:
        new_colors = []
        new_sizes = []
        for reg in filtered_df["Region"]:
            if reg == highlighted_region:
                new_colors.append("red")
                new_sizes.append(12)
            else:
                new_colors.append("#52b69a")
                new_sizes.append(7)
        fig_line.data[0].marker.color = new_colors
        fig_line.data[0].marker.size = new_sizes

    # Info for the markdown section
    info_text = f"**Filtering Unemployment From:** {low:.0f}% to {high:.0f}%.  \n\n"
    info_text += "**Regions Shown:**  \n"
    if len(filtered_df) > 0:
        for reg in filtered_df["Region"]:
            info_text += f"- {reg}  \n"
    else:
        info_text += "(No regions match current filters)  \n"

    info_text += "  \n"  # Extra spacing

    # Selection region info or average by filter
    if len(filtered_df) > 0:
        if highlighted_region and (highlighted_region in filtered_df["Region"].values):
            row = filtered_df[filtered_df["Region"] == highlighted_region].iloc[0]
            info_text += f"**Selected Region (Clicked):** {highlighted_region}  \n"
            info_text += f"**Unemployment Rate:** {row['Unemployment Rate']:.2f}%  \n"
            info_text += f"**Average Income:** {row['Average Income']:.2f}  \n"
            info_text += f"**Average Expenditure:** {row['Average Expenditure']:.2f}  \n"
        else:
            avg_unemp = filtered_df["Unemployment Rate"].mean()
            avg_income = filtered_df["Average Income"].mean()
            avg_expenditure = filtered_df["Average Expenditure"].mean()
            info_text += f"**Number of Regions in Filter:** {len(filtered_df)}  \n"
            info_text += f"**Avg. Unemployment Rate:** {avg_unemp:.2f}%  \n"
            info_text += f"**Avg. Income:** {avg_income:.2f}  \n"
            info_text += f"**Avg. Expenditure:** {avg_expenditure:.2f}  \n"

    md_component = dcc.Markdown(info_text)

    return fig_bar, fig_line, md_component

if __name__ == "__main__":
    pass

server = app.server