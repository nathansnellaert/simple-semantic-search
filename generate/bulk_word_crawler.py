import argparse
import json
import io
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def read_keywords(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def create_batch_calls(keywords):
    batch_calls = []
    for word in keywords:
        prompt = f'Given the word "{word}", provide the top 5 related words as a JSON object, where the key is "keywords", and the value is a list of lowercase keywords.'
        messages = [{"role": "user", "content": prompt}]
        request = {"custom_id": word, "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": messages, "max_tokens": 250, "temperature": 0}}
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
    graph = {}
    lines = content.decode().split('\n')
    for line in lines:
        if not line.strip():
            continue
        response = json.loads(line)
        custom_id = response["custom_id"]
        completion = response['response']["body"]["choices"][0]["message"]["content"]
        if '```json' in completion:
            completion = completion.split("```json\n")[1].split("\n```")[0]
        try:
            body = json.loads(completion)
            value = body['keywords']
            graph[custom_id] = value
        except:
            print(f"Error parsing JSON for custom_id: {custom_id}")
            print(completion)
    return graph

def main():
    parser = argparse.ArgumentParser(description="Process keywords and submit OpenAI batch job")
    parser.add_argument("input_file", help="Path to the txt file containing keywords")
    parser.add_argument("job_id", help="Unique identifier for this job")
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
            graph = process_output(content)
            with open(f"{args.job_id}.json", 'w') as file:
                json.dump(graph, file, indent=2)
            print(f"Graph saved to {args.job_id}.json")
            return
        else:
            print(f"Job ended with status: {batch_status.status}. Exiting.")
            return
    
    # If job doesn't exist, create and submit it
    keywords = read_keywords(args.input_file)
    batch_calls = create_batch_calls(keywords)
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
            "description": f"Generate related words for {len(keywords)} keywords"
        }
    )

    print(f"Batch job submitted successfully with ID: {args.job_id}")
    print("You can run this script again with the same job ID to check the status and process the results.")

if __name__ == "__main__":
    main()