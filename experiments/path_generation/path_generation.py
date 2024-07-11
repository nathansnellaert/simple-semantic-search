import os
import json
import io
import time
import argparse
import random
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import random
import numpy as np

def read_keywords(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def create_batch_calls(word_pairs, model, max_tokens):
    batch_calls = []
    for start_word, end_word, path_length in word_pairs:
        prompt = f'Generate a path of exactly {path_length} related words connecting "{start_word}" to "{end_word}". Provide the result as a JSON object with a key "path" containing a list of {path_length} words, including the start and end words.'
        messages = [{"role": "user", "content": prompt}]
        
        request = {
            "custom_id": f"{start_word}_{end_word}_{path_length}_{random.randint(1000, 9999)}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
                "response_format": {"type": "json_object"}
            }
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
    graph = []
    lines = content.decode().split('\n')
    for line in lines:
        if not line.strip():
            continue
        response = json.loads(line)
        custom_id = response["custom_id"]
        completion = response['response']["body"]["choices"][0]["message"]["content"]
        loaded = json.loads(completion)
        graph.append(loaded)
        
       
    return graph

def calculate_similarity(word1, word2, sentence_transformer):
    embedding1 = sentence_transformer.encode(word1, convert_to_tensor=True)
    embedding2 = sentence_transformer.encode(word2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    return similarity

def discretize_similarity(similarity):
    mean_similarity = 0.2
    std_similarity = 0.6
    target_mean = 5
    target_std = 1.5

    z_score = (similarity - mean_similarity) / std_similarity

    path_length = round(target_mean + (z_score * target_std))

    return max(1, min(10, path_length))

def discretize_similarity_percentile(similarity, similarity_distribution):
    min_path_length = 1
    max_path_length = 10
    
    percentile = np.percentile(similarity_distribution, similarity * 100)
    
    path_length = max_path_length - (percentile / 100) * (max_path_length - min_path_length)    
    return round(max(min_path_length, min(max_path_length, path_length)))

def transform_to_graph(data):
    graph = {}
    
    for item in data:
        path = item['path']
        for i, node in enumerate(path):
            if node not in graph:
                graph[node] = []
            
            if i > 0 and path[i-1] not in graph[node]:
                graph[node].append(path[i-1])
            
            if i < len(path) - 1 and path[i+1] not in graph[node]:
                graph[node].append(path[i+1])
    
    return graph

def main():
    parser = argparse.ArgumentParser(description="Build a word path graph using GPT-4 and OpenAI batch API")
    parser.add_argument("--word_list_file", required=True, help="File containing the list of words to use")
    parser.add_argument("--n_paths", type=int, default=1000, help="Number of paths to sample")
    parser.add_argument("--max_tokens", type=int, default=300, help="Maximum tokens for GPT-4 response")
    parser.add_argument("--model", default="gpt-4o", help="GPT model to use")
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
            paths_data = process_output(content)
            
            # Save paths
            with open(f"{args.job_id}_paths.json", 'w') as file:
                json.dump(paths_data, file, indent=2)
            print(f"Paths saved to {args.job_id}_paths.json")
            
            # Transform to graph and save
            graph = transform_to_graph(paths_data)
            with open(f"{args.job_id}_graph.json", 'w') as file:
                json.dump(graph, file, indent=2)
            print(f"Graph saved to {args.job_id}_graph.json")
            
            return
        else:
            print(f"Job ended with status: {batch_status.status}. Exiting.")
            return
    
    # If job doesn't exist, create and submit it
    word_list = read_keywords(args.word_list_file)
    sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    word_pairs = []
    for _ in range(args.n_paths):
        start_word, end_word = random.sample(word_list, 2)
        similarity = calculate_similarity(start_word, end_word, sentence_transformer)
        path_length = discretize_similarity(similarity)
        word_pairs.append((start_word, end_word, path_length))

    batch_calls = create_batch_calls(word_pairs, args.model, args.max_tokens)
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
            "description": f"Generate word paths for {args.n_paths} word pairs"
        }
    )

    print(f"Batch job submitted successfully with ID: {args.job_id}")
    print("You can run this script again with the same job ID to check the status and process the results.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()