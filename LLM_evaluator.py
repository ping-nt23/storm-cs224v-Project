import csv
import json
import os
import argparse
from pydantic import BaseModel, Field
from together import Together

together = Together()

# Define the schema for the output
class ScoreOutput(BaseModel):
    score: int = Field(description="A score from 1 to 5, 1 being the worst and 5 being the best based on the given criteria")
    reason: str = Field(description="A reason justifying the given score")

def main(input_file, output_file):
    # Open the CSV file and read the content
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        all_outputs = []

        # Loop over each row in the CSV
        for row in reader:
            transcript = f"QUERY: {row['QUERY']}; ARTICLE: {row['ARTICLE']}"
            # Call the LLM with the JSON schema
            print(transcript)
            extract = together.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """
                        Given the following QUERY, critically review the article and rate it on a scale from 1 to 5, where 1 is below average and 5 is outstanding. Be extremely strict and very critical in your evaluation. Focus on how closely the article resembles a “recent news” piece about QUERY, assuming the reader is already familiar with the context or doesn’t require excessive background information. Thus, the article should not read like a Wikipedia article or a text from an encyclopedia. If it does, it should automatically score a 1 or 2. The article should primarily contain timely, relevant, and concise information related to QUERY, just as you would expect from a professional recent news article.
Evaluate the article based on the following three criteria:
1. Format and Voice:
Does the article adhere to a concise, factual 'recent news' format, focusing only on timely information about QUERY? Avoids unnecessary details or background information unless directly relevant to current events.
Score 5: The article is extremely focused on recent, relevant information, without veering into unnecessary historical or background content at all. It feels like a proper news article.
Score 3: The article includes a lot of relevant current content but sometimes into background details.
Score 1: The article contains irrelevant or outdated information, making it feel more like a history lesson, background information, or definition than recent news.
2. Balancedness:
Does the article give fair coverage to all relevant points without overemphasizing one detail? For instance, if there are multiple recent events related to QUERY, they should be given appropriate attention in proportion to their significance.
Score 5: The article is extremely well-balanced, giving proportional attention to all important recent developments related to QUERY.
Score 3: The article provides a reasonable view of QUERY but focuses slightly too much on one aspect of QUERY at the expense of others.
Score 1: The article is overly focused on a single event or viewpoint of QUERY.
3. Relevance to Topic:
Is the article entirely focused on QUERY, without straying into unrelated areas? It should maintain a strong focus on the subject matter, without tangents.
Score 5: The article stays laser-focused on QUERY, with no irrelevant diversions.
Score 3: The article generally stays on-topic but includes a few minor tangents.
Score 1: The article sometimes drifts away from QUERY into unrelated topics.
Only answer in JSON.
                        """,
                    },
                    {
                        "role": "user",
                        "content": transcript,
                    },
                ],
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                response_format={
                    "type": "json_object",
                    "schema": ScoreOutput.model_json_schema(),
                },
            )
            output = json.loads(extract.choices[0].message.content)
            # Append the output to a list to save later
            all_outputs.append({
                "QUERY": row['QUERY'],
                "ARTICLE": row['ARTICLE'],
                "score_output": output
            })

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save all outputs to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)

    print(f"LLM output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a CSV file and generate scores based on its content.')
    parser.add_argument('filename', type=str, help='The input CSV file name (without extension)')
    args = parser.parse_args()

    # Construct the input and output file paths
    input_file = os.path.join('articles_input', f'{args.filename}.csv')
    output_file = os.path.join('LLM_eval_res', f'temp_{args.filename}_output.json')

    main(input_file, output_file)