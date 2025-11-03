#!/usr/bin/env python3
"""
JustGiving Data Collection Script
Recursive, fault-tolerant data collection system for gathering crowdfunding data from the JustGiving API.

This script implements a breadth-first search strategy with adaptive query refinement to maximize data coverage
while handling API limitations. It uses exponential backoff for retryable errors and maintains state for resume capability.
"""

import sys
import requests
import time
import json
import psycopg2
import string
from psycopg2.extras import execute_values
import os

# Configuration
APP_ID = 'YOUR_JUSTGIVING_APP_ID'  # Replace with your actual APP_ID
BASE_URL = f'https://api.justgiving.com/{APP_ID}'
PAGE_SIZE = 100
MAX_PAGES = 500
MAX_PAGES_THRESHOLD = 300
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0
MAX_RECURSION_DEPTH = 3
LETTERS = string.ascii_lowercase

# PostgreSQL connection
# Note: Set these environment variables or modify the connection parameters below
conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "YOUR_HOST"),
    database=os.getenv("DB_NAME", "YOUR_DATABASE_NAME"),
    user=os.getenv("DB_USER", "YOUR_USERNAME"),
    password=os.getenv("DB_PASSWORD", "YOUR_PASSWORD")
)
conn.autocommit = True
cursor = conn.cursor()

# Ensure necessary tables exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS query_state (
        query TEXT PRIMARY KEY,
        depth INT,
        current_page INT,
        fully_processed BOOLEAN DEFAULT FALSE
    );
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS crowdfunding (
        short_name TEXT PRIMARY KEY,
        details JSONB
    );
""")

def reset_query_state():
    """
    Truncate only the 'query_state' table,
    leaving existing records in 'crowdfunding' intact.
    """
    print("[RESET] Truncating table 'query_state' only. 'crowdfunding' remains unchanged.")
    cursor.execute("TRUNCATE TABLE query_state;")
    print("[RESET] Query state truncated. Next run will start from scratch for all queries.")

def request_with_retries(url, headers=None, params=None):
    """Perform a GET request with exponential backoff for retryable errors."""
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES+1):
        try:
            print(f"[HTTP] Attempt {attempt}: GET {url} with params={params}")
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            if response.status_code == 200:
                print("[HTTP] 200 OK - returning response.")
                return response
            
            elif response.status_code in (429, 503):
                print(f"[HTTP] {response.status_code} - Rate limit or temporary issue. Backing off {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= 2
                continue  # retry
                
            elif response.status_code in (401, 403):
                # Unlikely if you have a valid APP_ID; treat as non-recoverable
                print(f"[HTTP] {response.status_code} - Unauthorized or forbidden. Cannot proceed.")
                return response
            
            elif 400 <= response.status_code < 500:
                # 4xx (client error): probably not retryable
                print(f"[HTTP] {response.status_code} Client Error - Non-recoverable. {response.text}")
                return response
            
            elif 500 <= response.status_code < 600:
                # 5xx (server error) - may or may not be retryable
                print(f"[HTTP] {response.status_code} Server Error. Backing off {backoff:.1f}s before retry.")
                time.sleep(backoff)
                backoff *= 2
                continue
            
            else:
                # Some other status code
                print(f"[HTTP] Non-recoverable status {response.status_code}.")
                return response
        
        except requests.exceptions.RequestException as e:
            print(f"[HTTP] RequestException on attempt {attempt}: {e}. Backing off {backoff:.1f}s...")
            time.sleep(backoff)
            backoff *= 2
    
    print("[HTTP] Max retries exceeded, returning None.")
    return None

def get_crowdfunding_pages(query, page=1, page_size=PAGE_SIZE):
    """Fetch a page of search results for a given query."""
    url = f'{BASE_URL}/v1/fundraising/search'
    params = {
        'q': query,
        'page': page,
        'pageSize': page_size,
    }
    headers = {'Accept': 'application/json'}

    response = request_with_retries(url, headers=headers, params=params)
    if not response:
        print(f"[SEARCH] No response for query='{query}', page={page}.")
        return None
    
    if response.status_code != 200:
        print(f"[SEARCH] Non-200 status {response.status_code} for query='{query}', page={page}.")
        return None
    
    try:
        data = response.json()
        return data
    except ValueError as e:
        print(f"[SEARCH] Error parsing JSON for query='{query}', page={page}: {e}")
        return None

def get_crowdfunding_page_details(page_short_name):
    """Fetch the details for an individual crowdfunding page."""
    url = f'{BASE_URL}/v1/fundraising/pages/{page_short_name}'
    headers = {'Accept': 'application/json'}

    response = request_with_retries(url, headers=headers)
    if not response:
        print(f"[DETAILS] No response for page='{page_short_name}'.")
        return None
    
    if response.status_code != 200:
        if response.status_code == 404:
            print(f"[DETAILS] Page '{page_short_name}' not found (404). Skipping.")
            return None
        print(f"[DETAILS] Non-recoverable error {response.status_code} for page '{page_short_name}': {response.text}")
        return None
    
    try:
        return response.json()
    except ValueError as e:
        print(f"[DETAILS] Error parsing JSON for page='{page_short_name}': {e}")
        return None

def extract_page_short_name(page_url):
    """Extract short name from a page URL."""
    if not page_url:
        return None
    return page_url.strip('/').split('/')[-1]

def save_data_to_db(page_short_name, page_details):
    """Save page data to the 'crowdfunding' table."""
    print(f"[DB] Saving '{page_short_name}' to the database (ON CONFLICT DO NOTHING).")
    cursor.execute("""
        INSERT INTO crowdfunding (short_name, details)
        VALUES (%s, %s)
        ON CONFLICT (short_name)
        DO NOTHING;
    """, (page_short_name, json.dumps(page_details)))

def update_query_state(query, depth, current_page=None, fully_processed=None):
    """Upsert the query state in the 'query_state' table."""
    if current_page is None:
        current_page = 1
    if fully_processed is None:
        fully_processed = False

    cursor.execute("""
        INSERT INTO query_state (query, depth, current_page, fully_processed)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (query)
        DO UPDATE SET
            depth = EXCLUDED.depth,
            current_page = EXCLUDED.current_page,
            fully_processed = EXCLUDED.fully_processed;
    """, (query, depth, current_page, fully_processed))

def get_query_state(query):
    cursor.execute("SELECT depth, current_page, fully_processed FROM query_state WHERE query = %s", (query,))
    return cursor.fetchone()

def get_unfinished_queries():
    cursor.execute("SELECT query, depth, current_page FROM query_state WHERE fully_processed = FALSE")
    return cursor.fetchall()

def download_data_for_query(query, depth):
    """
    For a given query, iterate through pages up to MAX_PAGES or totalPages,
    download each page of results, and store them. Return the total_pages found.
    """
    state = get_query_state(query)
    if state is not None:
        stored_depth, start_page, done = state
        # If query was fully_processed before, return immediately
        if done:
            print(f"[QUERY] Query '{query}' is already marked fully processed. Skipping.")
            return 0
        current_page = start_page if start_page else 1
        if depth < stored_depth:
            # Possibly an older depth? We'll trust the DB's depth
            depth = stored_depth
    else:
        current_page = 1
        update_query_state(query, depth, current_page)

    total_pages = None
    print(f"[QUERY] Processing query='{query}' at depth={depth} starting from page={current_page}")

    while True:
        if current_page > MAX_PAGES:
            print(f"[QUERY] Reached MAX_PAGES={MAX_PAGES} limit for query='{query}' at page={current_page}.")
            break

        print(f"[QUERY] Fetching search results for query='{query}', page={current_page}")
        search_results = get_crowdfunding_pages(query=query, page=current_page, page_size=PAGE_SIZE)
        if not search_results or 'SearchResults' not in search_results:
            print(f"[QUERY] No valid search results for query='{query}', page={current_page}. Stopping.")
            break

        fundraising_pages = search_results['SearchResults']
        if not fundraising_pages:
            print(f"[QUERY] SearchResults is empty for query='{query}', page={current_page}. Stopping.")
            break

        # Set total_pages once
        if total_pages is None:
            total_pages = search_results.get('totalPages', 1)
            print(f"[QUERY] totalPages={total_pages} for query='{query}'")

        # Process each page
        for page_data in fundraising_pages:
            page_url = page_data.get('PageUrl')
            page_short_name = extract_page_short_name(page_url)
            if not page_short_name:
                print("[QUERY] No valid PageUrl found or empty short_name.")
                continue

            # Check if already in DB
            cursor.execute("SELECT 1 FROM crowdfunding WHERE short_name = %s", (page_short_name,))
            if cursor.fetchone():
                print(f"[QUERY] Page '{page_short_name}' already in DB. Skipping.")
                continue

            details = get_crowdfunding_page_details(page_short_name)
            if details:
                save_data_to_db(page_short_name, details)

        # Update query state after successfully processing this page
        update_query_state(query, depth, current_page + 1)
        current_page += 1

        # If we've reached the last page, stop
        if current_page > total_pages:
            print(f"[QUERY] Processed all {total_pages} pages for query='{query}'.")
            break

    # Mark query as fully processed
    update_query_state(query, depth, fully_processed=True)
    print(f"[QUERY] Marking query='{query}' as fully processed.")

    return total_pages if total_pages else 0

def refine_query(query):
    return [query + letter for letter in string.ascii_lowercase]

def get_all_data(query, depth=1):
    """
    Downloads data for `query`, then if it exceeds MAX_PAGES_THRESHOLD, refines further.
    """
    total_pages = download_data_for_query(query, depth) or 0

    # If the search results are too large, refine the query if we haven't hit max recursion depth
    if total_pages >= MAX_PAGES_THRESHOLD and depth < MAX_RECURSION_DEPTH:
        print(f"[REFINE] Query='{query}' has {total_pages} pages >= threshold {MAX_PAGES_THRESHOLD}. Refining...")
        for q in refine_query(query):
            get_all_data(q, depth+1)

def main():
    """Main entry point for the script."""
    # Check for --reset-state argument (but do NOT truncate 'crowdfunding')
    if '--reset-state' in sys.argv:
        reset_query_state()

    # Resume from unfinished queries if any
    unfinished = get_unfinished_queries()
    if unfinished:
        print("[MAIN] Resuming unfinished queries...")
        for q, d, p in unfinished:
            print(f"   -> Resuming query='{q}', depth={d}, current_page={p}")
            get_all_data(q, d)
    else:
        print("[MAIN] No unfinished queries found. Starting fresh with single-letter queries.")
        for letter in LETTERS:
            print(f"   -> Processing query='{letter}' at depth=1")
            get_all_data(letter, depth=1)

    print("[MAIN] All queries completed (or none to process). Exiting.")

if __name__ == '__main__':
    main()

