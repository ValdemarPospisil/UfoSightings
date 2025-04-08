import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import re
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Connect to DuckDB database
con = duckdb.connect("ufo.db")

print("\n🛸 UFO SIGHTINGS DATA MINING ANALYSIS 🛸\n")
print("Provádím analýzu pomocí technik data miningu na datech o UFO...\n")

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
    d.ufo_shape
FROM fact_sightings f
JOIN dim_location l ON f.location_id = l.location_id
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

# Query for sightings with descriptions for text mining
text_query = """
SELECT 
    f.sighting_id,
    d.ufo_shape,
    f.description
FROM fact_sightings f
JOIN dim_ufo d ON f.ufo_id = d.ufo_id
WHERE 
    f.description IS NOT NULL AND 
    LENGTH(f.description) > 50 AND
    d.ufo_shape IS NOT NULL AND
    d.ufo_shape != 'unknown'
LIMIT 5000
"""

# Query for prediction model data (predict whether sighting will be longer than 5 minutes)
prediction_query = """
SELECT 
    f.length_of_encounter_seconds > 300 AS long_sighting,
    d.ufo_shape,
    t.hour,
    t.is_weekend,
    t.season,
    l.country_code,
    SUBSTR(l.locale, 1, 50) AS locale
FROM fact_sightings f
JOIN dim_location l ON f.location_id = l.location_id
JOIN dim_time t ON f.time_id = t.time_id
JOIN dim_ufo d ON f.ufo_id = d.ufo_id
WHERE 
    f.length_of_encounter_seconds IS NOT NULL AND
    d.ufo_shape IS NOT NULL AND
    d.ufo_shape != 'unknown' AND
    t.hour IS NOT NULL AND
    l.country_code IS NOT NULL
LIMIT 10000
"""

# Load data into DataFrames
location_df = con.execute(location_query).fetchdf()
text_df = con.execute(text_query).fetchdf()
prediction_df = con.execute(prediction_query).fetchdf()

print(f"Načteno {len(location_df)} záznamů pro shlukování")
print(f"Načteno {len(text_df)} záznamů pro text mining")
print(f"Načteno {len(prediction_df)} záznamů pro klasifikaci\n")

# =============================================
# Part 2: Clustering (K-means)
# =============================================

print("Provádím shlukovou analýzu (K-means clustering)...")

# Prepare data for clustering
cluster_data = location_df[['latitude', 'longitude', 'length_of_encounter_seconds']].copy()

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
location_df['cluster'] = clusters

# Analyze clusters
cluster_stats = location_df.groupby('cluster').agg({
    'latitude': 'mean',
    'longitude': 'mean',
    'length_of_encounter_seconds': ['mean', 'median', 'count'],
    'ufo_shape': lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else 'N/A'
}).reset_index()

# Flatten multi-level columns
cluster_stats.columns = ['cluster', 'avg_latitude', 'avg_longitude', 
                         'avg_duration_seconds', 'median_duration_seconds', 
                         'count', 'most_common_shape']

# Convert seconds to minutes for readability
cluster_stats['avg_duration_minutes'] = cluster_stats['avg_duration_seconds'] / 60
cluster_stats['median_duration_minutes'] = cluster_stats['median_duration_seconds'] / 60

# Print cluster statistics
print("\nVýsledky shlukové analýzy:")
print(tabulate(cluster_stats[['cluster', 'count', 'avg_latitude', 'avg_longitude', 
                              'avg_duration_minutes', 'median_duration_minutes', 
                              'most_common_shape']], 
               headers=['Cluster', 'Počet', 'Průměrná šířka', 'Průměrná délka', 
                        'Prům. doba (min)', 'Medián doby (min)', 'Nejčastější tvar'],
               tablefmt="fancy_grid", 
               floatfmt=".2f",
               showindex=False))

# Visualize clusters on a map
plt.figure(figsize=(12, 8))
scatter = plt.scatter(location_df['longitude'], location_df['latitude'], 
                     c=location_df['cluster'], cmap='viridis', 
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
plt.savefig("ufo_clusters_map.png")

# Visualize inertia (elbow method)
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Počet shluků (k)', fontsize=14)
plt.ylabel('Inertia', fontsize=14)
plt.title('Elbow Method pro určení optimálního počtu shluků', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("ufo_elbow_method.png")

print("Vizualizace shluků uložena jako 'ufo_clusters_map.png'")
print("Graf 'elbow method' uložen jako 'ufo_elbow_method.png'\n")

# =============================================
# Part 3: Text Mining
# =============================================

print("Provádím analýzu textu (text mining) na popisech UFO pozorování...")

# Clean text function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply text cleaning
text_df['cleaned_text'] = text_df['description'].apply(clean_text)

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=1000,
    min_df=5,
    max_df=0.7,
    stop_words='english'
)

# Create document-term matrix
tfidf_matrix = tfidf.fit_transform(text_df['cleaned_text'])

# Get feature names
feature_names = np.array(tfidf.get_feature_names_out())

# Dimensionality reduction for visualization
svd = TruncatedSVD(n_components=2, random_state=42)
reduced_features = svd.fit_transform(tfidf_matrix)

# Add reduced features and shape to dataframe
text_df['x'] = reduced_features[:, 0]
text_df['y'] = reduced_features[:, 1]

# Get most important terms for each UFO shape
shape_terms = {}
for shape in text_df['ufo_shape'].unique():
    # Get indices for this shape
    shape_indices = text_df[text_df['ufo_shape'] == shape].index
    
    if len(shape_indices) < 5:  # Skip shapes with too few samples
        continue
        
    # Get the TF-IDF vectors for this shape
    shape_vectors = tfidf_matrix[shape_indices]
    
    # Sum TF-IDF values across all documents of this shape
    shape_importance = np.asarray(shape_vectors.sum(axis=0)).flatten()
    
    # Get indices of top terms
    top_indices = shape_importance.argsort()[-10:][::-1]
    
    # Get top terms and their scores
    top_terms = [(feature_names[i], shape_importance[i]) for i in top_indices]
    shape_terms[shape] = top_terms

# Print top terms by shape
print("\nNejdůležitější slova v popisech podle tvaru UFO:")
for shape, terms in shape_terms.items():
    print(f"\n{shape.upper()}:")
    term_table = [(term, score) for term, score in terms]
