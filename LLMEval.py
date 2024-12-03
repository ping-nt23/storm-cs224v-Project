import csv
import json
import os
import argparse
from pydantic import BaseModel, Field
from together import Together

together = Together()

# Define the schema for the output
class ScoreOutput(BaseModel):
    relevance: dict = Field(description="A score from 1 to 5 with a reason")
    conciseness: dict = Field(description="A score from 1 to 5 with a reason")
    balanced_coverage: dict = Field(description="A score from 1 to 5 with a reason")

def main(input_file, output_file):
    # Open the CSV file and read the content
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        all_outputs = []

        # Loop over each row in the CSV
        for row in reader:
            transcript = f"QUERY: {row['QUERY']}; ARTICLE: {row['ARTICLE']}"
            
            # Call the LLM with the JSON schema
            print(f"Evaluating: {row['QUERY']}...")
            extract = together.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """
You are a critical evaluator for STORMReport, a service that provides concise, fact-driven, and critical summaries about the developments of specific topics. Your task is to rate the quality of an article in response to a given QUERY based on the following criteria. Assume the user wants a TL;DR-style article to get up to speed quickly without irrelevant details or distracting language.

**Evaluation Criteria:**

1. **Relevance to QUERY**:
   - The article should stay entirely focused on the QUERY, providing only the most relevant facts and developments. Background information or context is acceptable **only if it directly enhances understanding of the recent developments**.
   - **Score 5**: Laser-focused on QUERY with relevant, concise updates and no tangents.
   - **Score 3**: Generally focused on QUERY but includes minor tangents or some unnecessary details.
   - **Score 1**: Drifts away from QUERY or reads like a general encyclopedia entry.

2. **Conciseness and Clarity**:
   - The article should deliver information in a concise and straightforward manner, avoiding overly verbose sentences, unnecessary "fluffy" language, or clickbait-style phrasing. 
   - **Score 5**: Highly concise and clear, presenting facts and developments without filler.
   - **Score 3**: Reasonably clear but includes occasional verbosity or stylistic fluff.
   - **Score 1**: Overly verbose or contains language aimed at evoking emotions rather than delivering facts.

3. **Balanced Coverage**:
   - The article should give proportional attention to key aspects of the QUERY without overemphasizing any single point. It should not neglect important developments.
   - **Score 5**: Balanced coverage of all major recent developments about the QUERY.
   - **Score 3**: Covers the main points but slightly overemphasizes or neglects some details.
   - **Score 1**: Overly focused on one detail or neglects critical information.

**Output Instructions:**
- Evaluate the article against these criteria and provide a score between 1 and 5 for each category.
- Justify your scores in a single sentence for each category.
- Use this JSON schema for your response:
{
  "relevance": { "score": 1-5, "reason": "..." },
  "conciseness": { "score": 1-5, "reason": "..." },
  "balanced_coverage": { "score": 1-5, "reason": "..." }
}
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
            try:
                output = json.loads(extract.choices[0].message.content)
                # Append the output to a list to save later
                all_outputs.append({
                    "QUERY": row['QUERY'],
                    "ARTICLE": row['ARTICLE'],
                    "score_output": output
                })
            except Exception as e:
                print(f"Error parsing output for QUERY {row['QUERY']}: {e}")
                continue

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
    input_file = os.path.join(f'{args.filename}')
    output_file = os.path.join('LLM_eval_res', f'temp_{args.filename}_output.json')

    main(input_file, output_file)
