import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Connect to DuckDB database
con = duckdb.connect("ufo.db")

# SQL query to get seasonal UFO hotspots by state
query = """
WITH state_population AS (
    -- Approximating population by state using distinct locale counts as a proxy
    SELECT 
        region, 
        COUNT(DISTINCT locale) AS locale_count
    FROM dim_location
    WHERE country_code = 'USA' AND region IS NOT NULL
    GROUP BY region
),
sightings_by_state_season AS (
    SELECT
        l.region,
        CASE 
            WHEN t.month IN (12, 1, 2) THEN 'Winter'
            WHEN t.month IN (3, 4, 5) THEN 'Spring'
            WHEN t.month IN (6, 7, 8) THEN 'Summer'
            WHEN t.month IN (9, 10, 11) THEN 'Fall'
        END AS season,
        COUNT(*) AS sightings_count
    FROM fact_sightings f
    JOIN dim_location l ON f.location_id = l.location_id
    JOIN dim_time t ON f.time_id = t.time_id
    WHERE l.country_code = 'USA' AND l.region IS NOT NULL
    GROUP BY l.region, season, month
)
SELECT
    s.region AS state,
    s.season,
    s.sightings_count,
    s.sightings_count::FLOAT / p.locale_count AS sightings_per_locale,
    p.locale_count
FROM sightings_by_state_season s
JOIN state_population p ON s.region = p.region
WHERE p.locale_count > 5  -- Filter out states with too few locales for reliable stats
ORDER BY sightings_per_locale DESC
LIMIT 20
"""

# Load results into pandas DataFrame
df = con.execute(query).fetchdf()

# Print table to terminal
print("\n🗺️ UFO SIGHTING HOTSPOTS BY STATE AND SEASON 🗺️\n")
print(tabulate(df, 
               headers=['State', 'Season', 'Sightings Count', 'Sightings per Locale', 'Locale Count'],
               tablefmt="fancy_grid", 
               floatfmt=".2f",
               showindex=False))

# Create visualization
plt.figure(figsize=(14, 10))

# Create a color palette based on seasons
season_colors = {'Winter': 'skyblue', 'Spring': 'yellowgreen', 
                 'Summer': 'tomato', 'Fall': 'darkorange'}

# Create the bubble chart
sns.scatterplot(
    data=df,
    x="state",
    y="sightings_per_locale",
    size="sightings_count",
    hue="season",
    palette=season_colors,
    sizes=(50, 1000),
    alpha=0.7
)

# Customize the chart
plt.title("UFO Sighting Hotspots by State and Season", fontsize=16)
plt.xlabel("State", fontsize=14)
plt.ylabel("Sightings per Locale (Normalized Activity)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=90)
plt.tight_layout()

# Save the visualization
plt.savefig("ufo_hotspots.png")
print("\nVisualization saved as 'ufo_hotspots.png'")

# Close connection
con.close()
