import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import folium
from folium.plugins import MarkerCluster

# Connect to DuckDB database
con = duckdb.connect("ufo.db")

print("\n🛸 UFO SIGHTINGS DATA MINING ANALYSIS 🛸\n")
print("Provádím analýzu pomocí shlukování a asociačních pravidel na datech o UFO...\n")

# =============================================
# Part 1: Get and prepare data
# =============================================

print("Načítám data z databáze...")

# Query for getting data for clustering by location and duration
location_query = """
SELECT 
    f.sighting_id,
    l.latitude,
    l.longitude,
    f.length_of_encounter_seconds,
    d.ufo_shape,
    t.hour,
    t.season,
    t.is_weekend,
    l.country,
    l.region
FROM fact_sightings f
JOIN dim_location l ON f.location_id = l.location_id
JOIN dim_time t ON f.time_id = t.time_id
JOIN dim_ufo d ON f.ufo_id = d.ufo_id
WHERE 
    l.latitude IS NOT NULL AND 
    l.longitude IS NOT NULL AND
    f.length_of_encounter_seconds IS NOT NULL AND
    f.length_of_encounter_seconds > 0 AND
    f.length_of_encounter_seconds < 3600 * 24 AND  -- Filter out sightings longer than 24 hours
    d.ufo_shape IS NOT NULL AND
    d.ufo_shape != 'unknown'
"""

# Load data into DataFrame
df = con.execute(location_query).fetchdf()

print(f"Načteno {len(df)} záznamů pro analýzu\n")

# =============================================
# Part 2: Clustering (K-means)
# =============================================

print("Provádím shlukovou analýzu (K-means clustering)...")

# Prepare data for clustering
cluster_data = df[['latitude', 'longitude', 'length_of_encounter_seconds']].copy()

# Log transform duration (since it's usually skewed)
cluster_data['log_duration'] = np.log1p(cluster_data['length_of_encounter_seconds'])

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_data[['latitude', 'longitude', 'log_duration']])

# Determine optimal number of clusters using inertia
inertia = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Select final number of clusters
k = 5  # You can adjust based on elbow plot
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to original data
df['cluster'] = clusters

# Analyze clusters
cluster_stats = df.groupby('cluster').agg({
    'latitude': 'mean',
    'longitude': 'mean',
    'length_of_encounter_seconds': ['mean', 'median', 'count'],
    'ufo_shape': lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else 'N/A',
    'hour': 'mean',
    'season': lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else 'N/A',
    'is_weekend': 'mean'
}).reset_index()

# Flatten multi-level columns
cluster_stats.columns = ['cluster', 'avg_latitude', 'avg_longitude', 
                         'avg_duration_seconds', 'median_duration_seconds', 
                         'count', 'most_common_shape', 'avg_hour', 
                         'most_common_season', 'weekend_ratio']

# Convert seconds to minutes for readability
cluster_stats['avg_duration_minutes'] = cluster_stats['avg_duration_seconds'] / 60
cluster_stats['median_duration_minutes'] = cluster_stats['median_duration_seconds'] / 60

# Print cluster statistics
print("\nVýsledky shlukové analýzy:")
print(tabulate(cluster_stats[['cluster', 'count', 'avg_latitude', 'avg_longitude', 
                              'avg_duration_minutes', 'median_duration_minutes', 
                              'most_common_shape', 'avg_hour', 'most_common_season', 'weekend_ratio']], 
               headers=['Cluster', 'Počet', 'Průměrná šířka', 'Průměrná délka', 
                        'Prům. doba (min)', 'Medián doby (min)', 'Nejčastější tvar',
                        'Prům. hodina', 'Nejčastější roční období', 'Poměr víkendů'],
               tablefmt="fancy_grid", 
               floatfmt=".2f",
               showindex=False))

# Visualize clusters on a map (static)
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['longitude'], df['latitude'], 
                     c=df['cluster'], cmap='viridis', 
                     alpha=0.6, s=30)

# Add cluster centers
centers = kmeans.cluster_centers_
plt.scatter(
    scaler.inverse_transform(centers)[:, 1],  # longitude is the second column (index 1)
    scaler.inverse_transform(centers)[:, 0],  # latitude is the first column (index 0)
    c='red', s=200, alpha=0.8, marker='X'
)

plt.colorbar(scatter, label='Klastr')
plt.title('Geografické shluky UFO pozorování', fontsize=16)
plt.xlabel('Zeměpisná délka', fontsize=14)
plt.ylabel('Zeměpisná šířka', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("images/ufo_clusters_map.png")

# Create interactive folium map
map_center = [df['latitude'].mean(), df['longitude'].mean()]
ufo_map = folium.Map(location=map_center, zoom_start=3)

# Add cluster markers
for idx, row in cluster_stats.iterrows():
    folium.CircleMarker(
        location=[row['avg_latitude'], row['avg_longitude']],
        radius=15,
        popup=f"Cluster {row['cluster']}<br>Count: {row['count']}<br>Shape: {row['most_common_shape']}",
        color='red',
        fill=True,
        fill_color='red'
    ).add_to(ufo_map)

# Add a MarkerCluster layer for all sightings
marker_cluster = MarkerCluster().add_to(ufo_map)

# Add markers for each sighting
for idx, row in df.sample(min(1000, len(df))).iterrows():  # Sample to avoid too many markers
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Shape: {row['ufo_shape']}<br>Duration: {row['length_of_encounter_seconds']/60:.1f} min",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(marker_cluster)

# Save the interactive map
ufo_map.save("images/ufo_interactive_map.html")

# Visualize inertia (elbow method)
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Počet shluků (k)', fontsize=14)
plt.ylabel('Inertia', fontsize=14)
plt.title('Elbow Method pro určení optimálního počtu shluků', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("images/ufo_elbow_method.png")

print("Vizualizace shluků uložena jako 'ufo_clusters_map.png'")
print("Interaktivní mapa uložena jako 'ufo_interactive_map.html'")
print("Graf 'elbow method' uložen jako 'ufo_elbow_method.png'\n")

# =============================================
# Part 3: Association Rule Mining
# =============================================

print("Provádím analýzu asociačních pravidel...")

# Prepare data for association rules
# We'll use these categorical features
categorical_features = ['ufo_shape', 'season', 'is_weekend']

# Add some discretized numerical features
df['duration_category'] = pd.cut(
    df['length_of_encounter_seconds'], 
    bins=[0, 60, 300, 900, 3600, 86400],
    labels=['<1min', '1-5min', '5-15min', '15min-1hr', '>1hr']
)

df['hour_category'] = pd.cut(
    df['hour'], 
    bins=[-0.1, 5.9, 11.9, 17.9, 23.9],
    labels=['night', 'morning', 'afternoon', 'evening']
)

categorical_features.extend(['duration_category', 'hour_category'])

# One-hot encode the categorical features
# This creates a DataFrame where each column is a specific feature value and contains 0 or 1
encoded_df = pd.get_dummies(df[categorical_features])

# Find frequent itemsets
min_support = 0.03  # Minimum support threshold - adjust as needed
frequent_itemsets = apriori(encoded_df, min_support=min_support, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# Generate association rules
min_threshold = 0.7  # Minimum confidence threshold - adjust as needed
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)

# Sort rules by lift (how much more likely the consequent given the antecedent vs in general)
rules = rules.sort_values('lift', ascending=False)

# Print the top association rules
print("\nNejsilnější asociační pravidla v datech o UFO:")
if len(rules) > 0:
    # Format the rules for better readability
    def format_itemset(itemset):
        return ', '.join([str(item) for item in itemset])
    
    rules_table = []
    for idx, row in rules.head(15).iterrows():
        antecedent = format_itemset(row['antecedents'])
        consequent = format_itemset(row['consequents'])
        rules_table.append([
            antecedent,
            consequent,
            row['support'],
            row['confidence'],
            row['lift']
        ])
    
    print(tabulate(
        rules_table,
        headers=['Pokud', 'Pak', 'Podpora', 'Spolehlivost', 'Lift'],
        tablefmt="fancy_grid",
        floatfmt=".3f"
    ))
    
    # Visualization of top 10 rules by lift
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=rules.head(20),
        x="support",
        y="confidence",
        size="lift",
        sizes=(100, 1000),
        hue="lift",
        palette="viridis",
        alpha=0.7
    )
    
    plt.title('Top 20 asociačních pravidel podle lift', fontsize=14)
    plt.xlabel('Podpora (Support)', fontsize=12)
    plt.ylabel('Spolehlivost (Confidence)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("images/ufo_association_rules.png")
    print("Vizualizace asociačních pravidel uložena jako 'ufo_association_rules.png'")
else:
    print("Nebyla nalezena žádná asociační pravidla s danými parametry. Zkuste snížit hodnotu min_threshold nebo min_support.")

# Visualize the most common combinations of features
if len(frequent_itemsets) > 0:
    # Filter itemsets with length 2 or more
    multi_itemsets = frequent_itemsets[frequent_itemsets['length'] >= 2]
    multi_itemsets = multi_itemsets.sort_values('support', ascending=False)
    
    if len(multi_itemsets) > 0:
        plt.figure(figsize=(12, 8))
        
        # Get top 15 itemsets by support
        top_itemsets = multi_itemsets.head(15)
        
        # Format itemset names for readability
        formatted_itemsets = [', '.join([str(item) for item in itemset]) for itemset in top_itemsets['itemsets']]
        
        # Create horizontal bar chart
        bars = plt.barh(formatted_itemsets, top_itemsets['support'], color='skyblue')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                    ha='left', va='center')
        
        plt.title('Nejčastější kombinace vlastností UFO pozorování', fontsize=14)
        plt.xlabel('Podpora (Support)', fontsize=12)
        plt.tight_layout()
        plt.savefig("images/ufo_frequent_itemsets.png")
        print("Vizualizace častých itemsetů uložena jako 'ufo_frequent_itemsets.png'")

print("\nAnalýza dokončena.")
