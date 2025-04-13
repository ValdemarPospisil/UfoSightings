import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Connect to DuckDB database
con = duckdb.connect("ufo.db")

# SQL query to get sightings by month and time of day
query = """
SELECT 
    CASE t.month
        WHEN 1 THEN 'January'
        WHEN 2 THEN 'February'
        WHEN 3 THEN 'March'
        WHEN 4 THEN 'April'
        WHEN 5 THEN 'May'
        WHEN 6 THEN 'June'
        WHEN 7 THEN 'July'
        WHEN 8 THEN 'August'
        WHEN 9 THEN 'September'
        WHEN 10 THEN 'October'
        WHEN 11 THEN 'November'
        WHEN 12 THEN 'December'
    END AS month,
    CASE
        WHEN t.hour >= 5 AND t.hour < 12 THEN 'Morning'
        WHEN t.hour >= 12 AND t.hour < 17 THEN 'Afternoon' 
        WHEN t.hour >= 17 AND t.hour < 22 THEN 'Evening'
        ELSE 'Night'
    END AS time_of_day,
    COUNT(*) AS sightings_count
FROM fact_sightings f
JOIN dim_time t ON f.time_id = t.time_id
GROUP BY t.month, time_of_day
ORDER BY 
    t.month,
    CASE 
        WHEN time_of_day = 'Morning' THEN 1
        WHEN time_of_day = 'Afternoon' THEN 2
        WHEN time_of_day = 'Evening' THEN 3
        WHEN time_of_day = 'Night' THEN 4
    END
"""

# Load results into pandas DataFrame
df = con.execute(query).fetchdf()


from tabulate import tabulate

# Seznam časových bloků
time_blocks = ['Morning', 'Afternoon', 'Evening', 'Night']

# Seznam měsíců pro správné řazení
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

# Dictionary pro uložení jednotlivých tabulek
tables = {}

# Naplň tabulky pro každý časový blok
for time in time_blocks:
    temp_df = df[df['time_of_day'] == time][['month', 'sightings_count']]
    temp_df = temp_df.set_index('month').reindex(month_order).fillna(0).astype(int)
    temp_df.columns = [time]  # přejmenuj sloupec na časový blok
    tables[time] = temp_df

# Spoj všechny 4 tabulky podle indexu (měsíců)
combined_table = pd.concat(tables.values(), axis=1)

# Vytiskni fancy tabulku
print("\n🛸 UFO SIGHTINGS BY MONTH AND TIME OF DAY 🛸\n")
print(tabulate(combined_table, headers="keys", tablefmt="fancy_grid"))

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(combined_table, annot=True, fmt="d", cmap="viridis", linewidths=.5)
plt.title("UFO Sightings by Month and Time of Day", fontsize=16)
plt.xlabel("Time of Day", fontsize=12)
plt.ylabel("Month", fontsize=12)
plt.tight_layout()
plt.savefig("images/ufo_sightings_by_time.png")
print("\nVisualization saved as 'ufo_sightings_by_time.png'")

# Close connection
con.close()
