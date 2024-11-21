import csv

def generate_query_article_csv(input_file, output_file, column_name):
    """
    Generate a CSV file with QUERY and ARTICLE format for a given column in the input CSV file.

    :param input_file: Path to the input CSV file
    :param output_file: Path to the output CSV file
    :param column_name: Column name containing articles
    """
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = ['QUERY', 'ARTICLE']
        with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                if row['Topic'] and row[column_name]:
                    writer.writerow({'QUERY': row['Topic'], 'ARTICLE': row[column_name]})

# Input file path
input_csv_file = 'input.csv'

# Output file paths for each column
output_tfidf_file = 'tfidf_query_article.csv'
output_embedding_file = 'embedding_query_article.csv'
output_llm_file = 'llm_query_article.csv'

# Generate CSV files for each column
generate_query_article_csv(input_csv_file, output_tfidf_file, 'TFIDF')
generate_query_article_csv(input_csv_file, output_embedding_file, 'EMBEDDING')
generate_query_article_csv(input_csv_file, output_llm_file, 'LLM')

print("CSV files generated successfully:")
print(f"- TFIDF: {output_tfidf_file}")
print(f"- EMBEDDING: {output_embedding_file}")
print(f"- LLM: {output_llm_file}")
