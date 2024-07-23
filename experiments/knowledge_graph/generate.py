import os
import json
import io
import time
import argparse
import random
from openai import OpenAI
from tqdm import tqdm
import csv

RELATIONSHIP_TYPES = [
    "is_a", "part_of", "has_property", "can", "synonym", "antonym", "causes",
    "used_for", "located_in", "time_related", "made_of", "produces", "requires",
    "instance_of", "associated_with", "derived_from", "measures", "opposite_of",
    "symbolizes", "precedes"
]

def read_keywords(file_path):
    with open(file_path, 'r') as file:
        return list(set([line.strip() for line in file if line.strip()]))

def create_batch_calls(words, model, max_tokens):
    batch_calls = []
    for word in words:
        prompt = f'''For the word "{word}", generate up to 15 semantic relationships with other words. 
        You are allowed to use the following relationship types: {', '.join(RELATIONSHIP_TYPES)}. 
        Note that these are general, in many cases, they don't apply to the specific word. 
        In many cases, there are also multiple results for one relationship type. 
        Provide the result as a CSV with trips (headers: word1, relation, word2)
        Use ```csv markup.
        '''
        messages = [{"role": "user", "content": prompt}]
        
        request = {
            "custom_id": f"{word}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
        }
        batch_calls.append(request)
    return batch_calls

def create_batch_file(batch_calls):
    buffer = io.BytesIO()
    for request in batch_calls:
        buffer.write((json.dumps(request) + "\n").encode('utf-8'))
    buffer.seek(0)
    return buffer

def get_batch_status(client, job_id):
    batches = client.batches.list(limit=100)
    for batch in batches:
        if batch.metadata.get('id') == job_id:
            return batch
    return None

def process_output(content):
    relationships = []
    lines = content.decode().split('\n')
    for line in lines:
        if not line.strip():
            continue
        try:
            response = json.loads(line)
            content = response['response']["body"]["choices"][0]["message"]["content"]
            
            # Try to extract CSV data if it's wrapped in code blocks
            csv_data = content
            if '```csv' in content:
                csv_data = content.split('```csv')[1].split('```')[0].strip()
            elif '```' in content:
                csv_data = content.split('```')[1].strip()
            
            # Remove any leading/trailing whitespace and skip empty lines
            csv_lines = [line.strip() for line in csv_data.split('\n') if line.strip()]
            
            # Reconstruct CSV string
            csv_string = '\n'.join(csv_lines)
            
            reader = csv.DictReader(io.StringIO(csv_string))
            for row in reader:
                if all(key in row for key in ['word1', 'relation', 'word2']):
                    relationships.append({
                        "word1": row['word1'].strip(),
                        "relation": row['relation'].strip(),
                        "word2": row['word2'].strip()
                    })
        except Exception as e:
            print(f"Error processing line: {e}")
            print(f"Problematic line: {line}")
            continue

    print(f"Total relationships extracted: {len(relationships)}")
    return relationships

def transform_to_graph(relationships):
    graph = {}
    for rel in relationships:
        word1, relation, word2 = rel['word1'], rel['relation'], rel['word2']
        if word1 not in graph:
            graph[word1] = {}
        if relation not in graph[word1]:
            graph[word1][relation] = []
        graph[word1][relation].append(word2)
    
    return graph

def main():
    parser = argparse.ArgumentParser(description="Build a semantic relationship graph using GPT-4 and OpenAI batch API")
    parser.add_argument("--word_list_file", required=True, help="File containing the list of words to use")
    parser.add_argument("--max_tokens", type=int, default=500, help="Maximum tokens for GPT-4 response")
    parser.add_argument("--model", default="gpt-4", help="GPT model to use")
    parser.add_argument("--job_id", required=True, help="Unique identifier for this job")
    args = parser.parse_args()

    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

    client = OpenAI()

    batch_status = get_batch_status(client, args.job_id)
    
    if batch_status:
        print(f"Job with ID {args.job_id} already exists. Current status: {batch_status.status}")
        
        if batch_status.status == 'failed':
            print("Job failed. Exiting.")
            return
        
        while batch_status.status not in ['completed', 'failed', 'cancelled']:
            print(f"Job in progress. Completed: {batch_status.request_counts.completed} out of {batch_status.request_counts.total}")
            time.sleep(30)
            batch_status = get_batch_status(client, args.job_id)
        
        if batch_status.status == 'completed':
            print("Job completed. Processing output...")
            content = client.files.content(batch_status.output_file_id).content
            relationships = process_output(content)
            
            # Save relationships
            with open(f"{args.job_id}_relationships.json", 'w') as file:
                json.dump(relationships, file, indent=2)
            print(f"Relationships saved to {args.job_id}_relationships.json")
            
            # Transform to graph and save
            graph = transform_to_graph(relationships)
            with open(f"{args.job_id}_graph.json", 'w') as file:
                json.dump(graph, file, indent=2)
            print(f"Graph saved to {args.job_id}_graph.json")
            
            return
        else:
            print(f"Job ended with status: {batch_status.status}. Exiting.")
            return
    
    # If job doesn't exist, create and submit it
    word_list = read_keywords(args.word_list_file)

    batch_calls = create_batch_calls(word_list, args.model, args.max_tokens)
    batch_file = create_batch_file(batch_calls)

    batch_input_file = client.files.create(
        file=batch_file,
        purpose="batch"
    )

    client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "id": args.job_id,
            "description": f"Generate semantic relationships for {len(word_list)} words"
        }
    )

    print(f"Batch job submitted successfully with ID: {args.job_id}")
    print("You can run this script again with the same job ID to check the status and process the results.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()