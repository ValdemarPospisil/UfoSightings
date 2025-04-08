import duckdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

# Připojení k DuckDB databázi
con = duckdb.connect("ufo.db")

# Upravený SQL dotaz
query = """
SELECT d.ufo_shape AS ufo_shape, COUNT(*) AS sightings_count
FROM fact_sightings f
JOIN dim_location l ON f.location_id = l.location_id
JOIN dim_ufo d ON f.ufo_id = d.ufo_id
WHERE l.country_code = 'USA'
GROUP BY d.ufo_shape
ORDER BY sightings_count DESC
LIMIT 10
"""

# Načtení výsledků do pandas DataFrame
df = con.execute(query).fetchdf()

# Výpis tabulky do terminálu pomocí tabulate
print(tabulate(df, headers="keys", tablefmt="fancy_grid"))

# Uložení grafu do souboru
plt.figure(figsize=(10, 6))
sns.barplot(data = df, x = "sightings_count", y = "ufo_shape", hue = "ufo_shape", palette="magma")
plt.title("Top 10 nejčastějších tvarů UFO v USA")
plt.xlabel("Počet pozorování")
plt.ylabel("Tvar UFO")
plt.tight_layout()
plt.savefig("top_ufo_shapes_usa.png")
