import pandas as pd
import os

# Load the dataset
file_path = 'articles.csv'
data = pd.read_csv(file_path)

# Select relevant columns for parsing
relevant_columns = ["Topic", "STORM", "GPT", "OUR-STORM"]

# Filter the DataFrame to include only these columns
parsed_data = data[relevant_columns]

# Define output directory
output_dir = 'parsed_articles'
os.makedirs(output_dir, exist_ok=True)

# Iterate over each source and save to separate CSV files
for source in ["STORM", "GPT", "OUR-STORM"]:
    # Create a new DataFrame for each source
    output_df = parsed_data[["Topic", source]].rename(columns={"Topic": "QUERY", source: "ARTICLE"})
    
    # Drop rows with missing articles
    output_df = output_df.dropna(subset=["ARTICLE"])
    
    # Save to a CSV file
    output_file = os.path.join(output_dir, f"{source}_articles.csv")
    output_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Saved {source} articles to {output_file}")
