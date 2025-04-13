import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Connect to DuckDB database
con = duckdb.connect("ufo.db")

# SQL query to get median duration by UFO shape
query = """
SELECT 
    d.ufo_shape,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.length_of_encounter_seconds)::INTEGER AS median_seconds,
    COUNT(*) AS sightings_count,
    AVG(f.length_of_encounter_seconds)::INTEGER AS avg_seconds
FROM fact_sightings f
JOIN dim_ufo d ON f.ufo_id = d.ufo_id
WHERE f.length_of_encounter_seconds IS NOT NULL AND f.length_of_encounter_seconds > 0
GROUP BY d.ufo_shape
HAVING COUNT(*) > 30
ORDER BY median_seconds DESC
LIMIT 10
"""

# Load results into pandas DataFrame
df = con.execute(query).fetchdf()

# Convert seconds to minutes for better readability
df['median_minutes'] = df['median_seconds'] / 60
df['avg_minutes'] = df['avg_seconds'] / 60

# Print table to terminal
print("\n⏱️ UFO ENCOUNTER DURATION BY SHAPE ⏱️\n")
print(tabulate(df[['ufo_shape', 'median_minutes', 'avg_minutes', 'sightings_count']], 
               headers=['UFO Shape', 'Median Duration (min)', 'Avg Duration (min)', 'Sightings Count'],
               tablefmt="fancy_grid", 
               floatfmt=".2f",
               showindex=False))

# Create visualization
plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x="median_minutes", 
    y="ufo_shape", 
    data=df, 
    palette="plasma",
    hue="sightings_count",
    dodge=False
)

# Add labels and title
plt.title("Median UFO Encounter Duration by Shape", fontsize=16)
plt.xlabel("Duration (minutes)", fontsize=14)
plt.ylabel("UFO Shape", fontsize=14)

# Add count annotations
for i, row in enumerate(df.itertuples()):
    plt.text(
        row.median_minutes + 0.5, 
        i, 
        f"{row.sightings_count} sightings", 
        va='center'
    )

plt.tight_layout()
plt.savefig("images/ufo_encounter_duration.png")
print("\nVisualization saved as 'ufo_encounter_duration.png'")

# Close connection
con.close()
