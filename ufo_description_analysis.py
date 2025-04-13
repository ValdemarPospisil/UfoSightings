import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import re
from collections import Counter

# Connect to DuckDB database
con = duckdb.connect("ufo.db")

# SQL query to get UFO shapes and descriptions
query = """
SELECT 
    d.ufo_shape,
    f.description
FROM fact_sightings f
JOIN dim_ufo d ON f.ufo_id = d.ufo_id
WHERE f.description IS NOT NULL 
  AND LENGTH(f.description) > 20
  AND d.ufo_shape IS NOT NULL
  AND d.ufo_shape != 'unknown'
LIMIT 10000  -- Limit to a reasonable number for analysis
"""

# Load results into pandas DataFrame
df = con.execute(query).fetchdf()

# Function to clean and tokenize text
def clean_text(text):
    if not isinstance(text, str):
        return []
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stop words (common words that don't add much meaning)
    stop_words = {'the', 'and', 'to', 'of', 'a', 'in', 'that', 'it', 'was', 'for', 
                  'on', 'is', 'with', 'as', 'at', 'by', 'an', 'this', 'i', 'my', 
                  'we', 'our', 'you', 'your', 'they', 'their', 'he', 'she', 'his', 
                  'her', 'its', 'from', 'or', 'which', 'me', 'him', 'them', 'were', 
                  'been', 'have', 'has', 'had', 'be', 'there', 'when', 'who', 'what', 
                  'where', 'why', 'how'}
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    return tokens

# Apply text cleaning to descriptions
df['tokens'] = df['description'].apply(clean_text)

# Get top 5 common words for each UFO shape
shape_words = {}
shape_counts = {}

for shape in df['ufo_shape'].unique():
    shape_df = df[df['ufo_shape'] == shape]
    shape_counts[shape] = len(shape_df)
    
    # Flatten all tokens for this shape
    all_tokens = [token for tokens_list in shape_df['tokens'] for token in tokens_list]
    
    # Count tokens
    word_counts = Counter(all_tokens)
    
    # Get top 5 words
    top_words = word_counts.most_common(5)
    shape_words[shape] = top_words

# Create a DataFrame for visualization
viz_data = []
for shape, words in shape_words.items():
    for word, count in words:
        viz_data.append({
            'ufo_shape': shape,
            'word': word,
            'count': count,
            'frequency': count / shape_counts[shape] if shape_counts[shape] > 0 else 0
        })

viz_df = pd.DataFrame(viz_data)

# Print table of top words by UFO shape
print("\n📝 TOP WORDS IN UFO SIGHTING DESCRIPTIONS BY SHAPE 📝\n")

# Group by shape and create a formatted table
table_data = []
for shape in sorted(shape_words.keys()):
    words_with_counts = [f"{word} ({count})" for word, count in shape_words[shape]]
    table_data.append([shape, shape_counts[shape], ', '.join(words_with_counts)])

print(tabulate(table_data, 
               headers=['UFO Shape', 'Sightings Count', 'Top 5 Words (count)'],
               tablefmt="fancy_grid",
               showindex=False))

# Create heatmap visualization of word frequencies across shapes
# Pivot data for heatmap
pivot_df = viz_df.pivot_table(
    index='ufo_shape', 
    columns='word', 
    values='frequency',
    fill_value=0
)

# Sort shapes by count
sorted_shapes = [shape for shape, _ in 
                 sorted(shape_counts.items(), key=lambda x: x[1], reverse=True)]
pivot_df = pivot_df.reindex(sorted_shapes)

# Get top 20 words overall for better visualization
top_words_overall = Counter([word for tokens_list in df['tokens'] for word in tokens_list]).most_common(20)
top_words_list = [word for word, _ in top_words_overall]

# Filter columns to top words only
pivot_df = pivot_df[[col for col in pivot_df.columns if col in top_words_list]]

# Create heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(pivot_df, cmap="viridis", annot=False, linewidths=.5)
plt.title("Word Frequency in UFO Sighting Descriptions by Shape", fontsize=16)
plt.xlabel("Words", fontsize=14)
plt.ylabel("UFO Shape", fontsize=14)
plt.tight_layout()
plt.savefig("images/ufo_description_analysis.png")
print("\nVisualization saved as 'ufo_description_analysis.png'")

# Create word cloud alternative visualization
try:
    from wordcloud import WordCloud
    
    # Create a figure with subplots for top 4 UFO shapes
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Get top 4 UFO shapes by count
    top_shapes = sorted(shape_counts.items(), key=lambda x: x[1], reverse=True)[:4]
    
    for i, (shape, _) in enumerate(top_shapes):
        # Get all words for this shape
        shape_df = df[df['ufo_shape'] == shape]
        all_tokens = [token for tokens_list in shape_df['tokens'] for token in tokens_list]
        
        # Create word frequency dictionary
        word_freq = Counter(all_tokens)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='black',
            colormap='viridis',
            max_words=50
        ).generate_from_frequencies(word_freq)
        
        # Plot
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f"Shape: {shape} ({shape_counts[shape]} sightings)", fontsize=14)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("images/ufo_shape_wordclouds.png")
    print("Word cloud visualization saved as 'ufo_shape_wordclouds.png'")
except ImportError:
    print("WordCloud package not found. Skipping word cloud visualization.")

# Close connection
con.close()
