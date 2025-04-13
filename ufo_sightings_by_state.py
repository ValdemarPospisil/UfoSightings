import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import plotly.express as px
import numpy as np

# Connect to DuckDB database
con = duckdb.connect("ufo.db")

# SQL query to get UFO sighting statistics by state
query = """
WITH state_sightings AS (
    SELECT
        l.region,
        f.length_of_encounter_seconds
    FROM fact_sightings f
    JOIN dim_location l ON f.location_id = l.location_id
    WHERE l.country_code = 'USA' AND l.region IS NOT NULL
)
SELECT
    region AS state,
    COUNT(*) AS sightings_count,
    AVG(length_of_encounter_seconds) AS avg_encounter_seconds,
    MEDIAN(length_of_encounter_seconds) AS median_encounter_seconds
FROM state_sightings
GROUP BY region
ORDER BY sightings_count DESC
"""

# Load results into pandas DataFrame
df = con.execute(query).fetchdf()

# Clean any potential NaN values
df = df.fillna(0)

# Print table to terminal
print("\n🛸 UFO SIGHTING STATISTICS BY STATE 🛸\n")
print(tabulate(df, 
               headers=['State', 'Sightings Count', 'Avg. Duration (sec)', 'Median Duration (sec)'],
               tablefmt="fancy_grid", 
               floatfmt=".2f",
               showindex=False))

# Create bar chart visualization
plt.figure(figsize=(15, 10))

# Sort states by sightings count
df_sorted = df.sort_values('sightings_count', ascending=False)

# Create the bar chart
sns.barplot(
    data=df_sorted,
    x='sightings_count',
    y='state',
    hue='state',
    palette='viridis'
)

# Customize the chart
plt.title("UFO Sightings by State", fontsize=16)
plt.xlabel("State", fontsize=14)
plt.ylabel("Number of Sightings", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the visualization
plt.savefig("images/ufo_sightings_by_state.png")
print("\nBar chart saved as 'ufo_sightings_by_state.png'")

# Create an interactive choropleth map using Plotly
# Get state abbreviations mapping for plotly
state_abbr = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC'
}

# Ensure state names are in the right format and add abbreviations
df['state'] = df['state'].str.title()
df['state_code'] = df['state'].map(state_abbr)

# Create choropleth map
fig = px.choropleth(
    df,
    locations='state_code',
    color='sightings_count',
    hover_name='state',
    locationmode='USA-states',
    scope="usa",
    color_continuous_scale="Viridis",
    hover_data={
        'sightings_count': True,
        'avg_encounter_seconds': ':.2f',
        'median_encounter_seconds': ':.2f',
        'state_code': False
    },
    labels={
        'sightings_count': 'UFO Sightings',
        'avg_encounter_seconds': 'Avg Duration (sec)',
        'median_encounter_seconds': 'Median Duration (sec)'
    },
    title='UFO Sightings by US State'
)

# Update layout
fig.update_layout(
    geo=dict(
        showlakes=True,
        lakecolor='rgb(255, 255, 255)'
    ),
    coloraxis_colorbar=dict(
        title='Number of Sightings'
    ),
    width=1000,
    height=600
)

# Save as interactive HTML
fig.write_html("html/ufo_sightings_map.html")
print("Interactive map saved as 'ufo_sightings_map.html'")

# Close connection
con.close()
