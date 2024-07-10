"""
This script generates a graph of related words using OpenAI's GPT-4 model, starting from a given word and expanding outward. 
For large graphs is it recommended to use bulk_word_crawler.py, as it uses the bulk completion endpoint to save 50%.
"""
import os
import json
import time
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
def get_gpt_response(client, prompt, model, max_tokens):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def get_related_words(client, word, n_related_words, model, max_tokens):
    prompt = f'Given the word "{word}", provide the top {n_related_words} related words as a JSON object, where the key is "keywords", and the value is a list of lowercase keywords.'
    response = get_gpt_response(client, prompt, model, max_tokens)
    return [w.lower() for w in response.get("keywords", []) if w.islower() and len(w.split()) == 1]

def save_state(word_graph, file_path, file_lock):
    with file_lock:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(word_graph, f, indent=2)

def load_state(file_path, file_lock):
    with file_lock:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
    return {}

def process_word(client, word, word_graph, n_related_words, model, max_tokens):
    if word not in word_graph:
        related_words = get_related_words(client, word, n_related_words, model, max_tokens)
        return word, related_words
    return word, None

def build_word_graph(client, start_word, max_workers, max_words, n_related_words, model, max_tokens, file_path):
    file_lock = Lock()
    word_graph = load_state(file_path, file_lock)
    crawled_words = set(word_graph.keys())
    all_words = set(word_graph.keys())

    for words in word_graph.values():
        all_words.update(words)
    
    to_crawl = deque(all_words - crawled_words)
    
    if not to_crawl and not word_graph:
        to_crawl.append(start_word)

    progress_bar = tqdm(total=max_words, initial=len(word_graph), desc="Processing words")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while to_crawl and len(word_graph) < max_words:
            batch = [to_crawl.popleft() for _ in range(min(max_workers, len(to_crawl), max_words - len(word_graph)))]
            futures = [executor.submit(process_word, client, word, word_graph, n_related_words, model, max_tokens) for word in batch]
            
            for future in as_completed(futures):
                word, related_words = future.result()
                if related_words:
                    word_graph[word] = related_words
                    progress_bar.update(1)
                    for related_word in related_words:
                        if related_word not in all_words and len(word_graph) < max_words:
                            to_crawl.append(related_word)
                            all_words.add(related_word)
            
            save_state(word_graph, file_path, file_lock)

    progress_bar.close()
    return word_graph

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build a word graph using GPT-4")
    parser.add_argument("--start_word", default="programming", help="The starting word for the graph")
    parser.add_argument("--max_workers", type=int, default=8, help="Maximum number of concurrent workers")
    parser.add_argument("--max_words", type=int, default=1000, help="Maximum number of words to process")
    parser.add_argument("--n_related_words", type=int, default=5, help="Number of related words to request for each word")
    parser.add_argument("--max_tokens", type=int, default=300, help="Maximum tokens for GPT-4 response")
    parser.add_argument("--model", default="gpt-4o", help="JSON compatible GPT model to use")
    parser.add_argument("--file_path", default="word_graph.json", help="File path for saving/loading state")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    start_time = time.time()

    word_graph = build_word_graph(
        client, 
        args.start_word, 
        args.max_workers, 
        args.max_words, 
        args.n_related_words, 
        args.model, 
        args.max_tokens, 
        args.file_path
    )
    end_time = time.time()

    print(f"\nProcess completed in {end_time - start_time:.2f} seconds")
    print(f"Total words processed: {len(word_graph)}")