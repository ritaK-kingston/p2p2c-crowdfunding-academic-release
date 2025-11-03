# JustGiving Data Collection Script

This script implements a sophisticated, recursive, and fault-tolerant data collection system for gathering crowdfunding campaign data from the JustGiving API. The system is designed to handle large-scale data collection while gracefully managing API limitations, rate limits, and network issues.

## Overview

The script uses a **breadth-first search strategy** with **adaptive query refinement** to maximize data coverage across the JustGiving platform. It implements exponential backoff for retryable errors, maintains comprehensive state tracking for resume capability, and includes duplicate prevention mechanisms to ensure efficient and reliable data collection.

### How It Works

The data collection process operates through a multi-level recursive strategy:

1. **Initial Query Generation**: The script starts with single-letter queries (a-z) representing the 26 possible first characters of campaign URLs or identifiers. This breadth-first approach ensures maximum coverage of available campaigns.

2. **API Interaction and Pagination**: For each query, the script fetches results from the JustGiving API, handling pagination automatically. It processes up to `MAX_PAGES` (default: 500) pages per query, with each page containing `PAGE_SIZE` (default: 100) results.

3. **Adaptive Query Refinement**: When a query returns a large number of results (≥`MAX_PAGES_THRESHOLD`, default: 300 pages), the system automatically refines the query into sub-queries. For example, if the query "c" returns too many results, it splits into "ca", "cb", "cc", ... "cz" (26 sub-queries). This refinement can continue recursively to depth 3 (e.g., "ca" → "caa", "cab", ... "caz").

4. **State Management**: The script maintains state in a PostgreSQL `query_state` table, tracking:
   - Current query being processed
   - Depth level (1, 2, or 3)
   - Current page number within the query
   - Whether the query has been fully processed

5. **Resume Capability**: If the script is interrupted (e.g., network failure, manual stop), it can be restarted and will automatically resume from unfinished queries stored in the state table. This eliminates the need to restart from scratch.

6. **Fault-Tolerant Error Handling**: The script implements sophisticated error handling:
   - **Rate Limits (429, 503)**: Automatically retries with exponential backoff
   - **Server Errors (5xx)**: Retries with exponential backoff
   - **Client Errors (4xx)**: Logs and continues (except authentication errors which stop execution)
   - **Network Issues**: Retries with exponential backoff

7. **Duplicate Prevention**: Before fetching detailed campaign information, the script checks if the campaign already exists in the database, avoiding redundant API calls and ensuring data integrity.

8. **Data Storage**: Complete campaign data is stored in PostgreSQL as JSONB format, providing flexibility for querying and extracting specific fields. The JSONB format allows storing the full API response while enabling efficient querying of specific fields.

## Key Features

- **Multi-Level Recursive Search Strategy**: Automatically refines queries to handle large result sets while ensuring comprehensive coverage
- **Fault-Tolerant HTTP Handling**: Exponential backoff retry logic for rate limits, server errors, and network issues ensures robust operation in real-world conditions
- **Resume Capability**: Comprehensive state tracking allows seamless resumption of interrupted collection sessions without losing progress
- **Duplicate Prevention**: Database checks before fetching prevent redundant API calls and ensure data integrity
- **Efficient Pagination**: Handles large result sets through systematic pagination processing
- **JSONB Storage**: Flexible PostgreSQL JSONB format enables efficient storage and querying of complex nested campaign data

## Configuration

Before running the script, you need to complete the following setup:

### 1. JustGiving API App ID

Obtain a JustGiving API App ID by registering at [JustGiving API](https://developers.justgiving.com/). The App ID is required for all API requests and should be included in request headers.

**Note**: The script includes a placeholder `YOUR_JUSTGIVING_APP_ID` that must be replaced with your actual App ID before running.

### 2. PostgreSQL Database Setup

Create a PostgreSQL database and configure the following environment variables or update the connection parameters in the script:

```bash
export DB_HOST=your_host        # e.g., localhost
export DB_NAME=your_database_name
export DB_USER=your_username
export DB_PASSWORD=your_password
```

Alternatively, you can modify the database connection parameters directly in the script if you prefer not to use environment variables.

### 3. Database Schema

The script automatically creates the required tables if they don't exist:

```sql
-- State management table for tracking query progress
CREATE TABLE query_state (
    query TEXT PRIMARY KEY,           -- The search query (e.g., "c", "ca", "catering")
    depth INT,                         -- Recursion depth (1, 2, or 3)
    current_page INT,                  -- Current page number being processed
    fully_processed BOOLEAN DEFAULT FALSE  -- Whether query has been fully processed
);

-- Data storage table for campaign information
CREATE TABLE crowdfunding (
    short_name TEXT PRIMARY KEY,       -- Unique campaign identifier from URL
    details JSONB                      -- Complete campaign data from API
);
```

The `query_state` table enables resume capability by tracking which queries have been processed and which are still pending. The `crowdfunding` table stores all collected campaign data in a flexible JSONB format.

## Usage

### Basic Usage

Run the script to begin data collection:

```bash
python justgiving_search.py
```

This will:
1. Initialize or resume from the `query_state` table
2. Start with single-letter queries (a-z) at depth 1
3. For each query, fetch all pages up to `MAX_PAGES` (500) or until no more pages are available
4. If a query returns ≥`MAX_PAGES_THRESHOLD` (300) pages, automatically refine it into sub-queries (e.g., "c" → "ca", "cb", ... "cz")
5. Store all campaign data in the `crowdfunding` table
6. Update the `query_state` table to track progress

### Resume Interrupted Collection

If the script is interrupted (e.g., network failure, manual stop, server restart), simply run it again:

```bash
python justgiving_search.py
```

The script will automatically detect unfinished queries in the `query_state` table and resume from where it left off. No data is lost, and previously collected campaigns are not re-fetched due to duplicate prevention.

### Reset Query State

To restart query processing from scratch (without deleting collected campaign data):

```bash
python justgiving_search.py --reset-state
```

This will clear the `query_state` table, allowing you to reprocess all queries from the beginning. **Note**: This does not delete collected campaign data from the `crowdfunding` table.

## Configuration Parameters

The script includes several configurable parameters that can be adjusted based on your needs:

- **`PAGE_SIZE`**: Number of results per API call (default: 100)
  - Adjust based on API limitations and desired granularity of progress tracking
- **`MAX_PAGES`**: Maximum pages to process per query (default: 500)
  - Total results per query = PAGE_SIZE × MAX_PAGES (default: 50,000 campaigns per query)
- **`MAX_PAGES_THRESHOLD`**: Threshold for query refinement (default: 300)
  - If a query returns this many pages or more, it will be refined into sub-queries
- **`MAX_RETRIES`**: Maximum retry attempts for failed requests (default: 5)
  - Each retry uses exponential backoff: INITIAL_BACKOFF × (2 ^ retry_number)
- **`INITIAL_BACKOFF`**: Initial backoff time in seconds (default: 2.0)
  - Doubles with each retry: 2s, 4s, 8s, 16s, 32s
- **`MAX_RECURSION_DEPTH`**: Maximum depth for query refinement (default: 3)
  - Depth 1: Single letters (a-z) → 26 queries
  - Depth 2: Letter pairs (aa-zz) → 26 sub-queries per depth-1 query
  - Depth 3: Letter triplets (aaa-zzz) → 26 sub-sub-queries per depth-2 query

## Query Refinement Strategy

The script implements an intelligent recursive refinement strategy to handle queries that return too many results:

### Depth 1: Single Letters
Initial queries consist of single letters (a-z):
- "a", "b", "c", ..., "z"
- 26 total queries

### Depth 2: Letter Pairs
If a depth-1 query returns ≥300 pages, it is refined into letter pairs:
- Example: "c" → "ca", "cb", "cc", ..., "cz"
- 26 sub-queries per refined query

### Depth 3: Letter Triplets
If a depth-2 query still returns ≥300 pages, it is refined into letter triplets:
- Example: "ca" → "caa", "cab", "cac", ..., "caz"
- 26 sub-sub-queries per refined query

### Example Refinement Process

```
Query "c" returns 400 pages
  → Refine into: ca, cb, cc, cd, ..., cz (26 sub-queries)
    Query "ca" returns 350 pages
      → Refine into: caa, cab, cac, ..., caz (26 sub-sub-queries)
    Query "cb" returns 250 pages
      → Process normally (no refinement needed)
```

This strategy ensures comprehensive coverage while respecting API pagination limits and avoiding queries that are too broad to process efficiently.

## Error Handling

The script implements comprehensive error handling for various HTTP status codes and network issues:

### Successful Responses
- **200 OK**: Success - proceed with data extraction and storage

### Retryable Errors (Exponential Backoff)
- **429 Too Many Requests**: Rate limit exceeded - wait and retry with exponential backoff
- **503 Service Unavailable**: Temporary server issue - retry with exponential backoff
- **5xx Server Errors**: Server-side issues - retry with exponential backoff
- **Network Timeouts**: Connection issues - retry with exponential backoff

### Non-Retryable Errors (Log and Continue)
- **404 Not Found**: Page/campaign not found - log and skip to next item
- **400 Bad Request**: Invalid request - log and skip (may indicate data issue)

### Fatal Errors (Stop Execution)
- **401 Unauthorized**: Authentication error - check API credentials
- **403 Forbidden**: Authorization error - check API permissions

All errors are logged with timestamps and relevant context for debugging.

## Data Output

Each campaign is stored in the `crowdfunding` table with the following structure:

- **`short_name`**: Unique identifier extracted from the campaign URL
  - Example: "johns-marathon-2024" from URL "https://www.justgiving.com/fundraising/johns-marathon-2024"
- **`details`**: Complete campaign data as JSONB, including:
  - Campaign story/narrative text
  - Metadata (creation date, activity type, etc.)
  - Fundraising statistics
  - All other fields returned by the JustGiving API

The JSONB format allows flexible querying:

```sql
-- Extract story text
SELECT details->>'story' FROM crowdfunding;

-- Filter by activity type
SELECT * FROM crowdfunding WHERE details->>'activityType' = 'InMemory';

-- Extract creation date
SELECT details->>'pageCreatedDate' FROM crowdfunding;
```

## Requirements

Install required packages:

```bash
pip install requests psycopg2-binary
```

Or use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Dependencies
- **`requests`**: HTTP library for API interactions with session management and retry capabilities
- **`psycopg2-binary`**: PostgreSQL database adapter for Python (binary wheel for easy installation)

## Implementation Details

### Duplicate Prevention

The script uses PostgreSQL's `ON CONFLICT DO NOTHING` clause to prevent duplicate entries:

```python
INSERT INTO crowdfunding (short_name, details)
VALUES (%s, %s)
ON CONFLICT (short_name) DO NOTHING;
```

This ensures that if a campaign is encountered multiple times (e.g., in overlapping query results), only one entry is stored.

### State Management

Query state is tracked separately from campaign data, allowing safe resumption without risk of data loss. The state table tracks:
- Which queries have been processed
- Current progress within each query (page number)
- Whether queries are fully complete

This separation enables:
- Safe interruption and resumption
- Progress tracking for long-running collection sessions
- Debugging and monitoring of collection progress

### Rate Limiting

The script respects API rate limits through:
- Exponential backoff on 429 (Too Many Requests) responses
- Configurable retry parameters
- Respect for API response headers (if available)

### Large-Scale Collection

For large-scale collection (hundreds of thousands of campaigns), the script is designed to:
- Run continuously for extended periods
- Handle interruptions gracefully
- Minimize redundant API calls
- Provide progress visibility through state tracking

## Notes

- The script uses `ON CONFLICT DO NOTHING` to prevent duplicate entries, ensuring data integrity
- Query state is tracked separately from campaign data, allowing safe resumption without risk of data loss
- The script respects API rate limits through exponential backoff, preventing API access issues
- Large queries are automatically refined to ensure comprehensive coverage while respecting API pagination limits
- The JSONB storage format provides flexibility for extracting and analyzing different aspects of the collected data
- The script can handle network interruptions, API rate limits, and server errors gracefully through retry logic

## License

Copyright 2024 [Author Names]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
