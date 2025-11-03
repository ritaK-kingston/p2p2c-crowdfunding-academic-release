# JustGiving Data Collection Script

This script implements a recursive, fault-tolerant data collection system for gathering crowdfunding data from the JustGiving API.

## Overview

The script uses a **breadth-first search strategy** with **adaptive query refinement** to maximize data coverage while handling API limitations. It implements exponential backoff for retryable errors and maintains state for resume capability.

## Key Features

- **Multi-Level Search Strategy**: Starts with single letters (a-z), then refines queries recursively if they return too many results
- **Fault-Tolerant HTTP Handling**: Exponential backoff retry logic for rate limits and server errors
- **Resume Capability**: Tracks query state in database to allow resuming interrupted collection
- **Duplicate Prevention**: Checks database before fetching page details to avoid redundant API calls

## Configuration

Before running, you need to:

1. **Get a JustGiving API App ID**: Register at [JustGiving API](https://developers.justgiving.com/) to obtain your `APP_ID`
2. **Set up PostgreSQL Database**: Create a database and set environment variables:
   ```bash
   export DB_HOST=your_host
   export DB_NAME=your_database_name
   export DB_USER=your_username
   export DB_PASSWORD=your_password
   ```
3. **Update the script**: Replace `YOUR_JUSTGIVING_APP_ID` with your actual App ID

## Database Schema

The script creates two tables:

```sql
-- State management table
CREATE TABLE query_state (
    query TEXT PRIMARY KEY,
    depth INT,
    current_page INT,
    fully_processed BOOLEAN DEFAULT FALSE
);

-- Data storage table
CREATE TABLE crowdfunding (
    short_name TEXT PRIMARY KEY,
    details JSONB
);
```

## Usage

### Basic Usage

```bash
python justgiving_search.py
```

This will:
- Start with single-letter queries (a-z) at depth 1
- For each query, fetch all pages up to MAX_PAGES (500) or until no more pages
- If a query returns ≥300 pages, refine it into sub-queries (e.g., "c" → "ca", "cb", ... "cz")
- Save all campaign data to the `crowdfunding` table

### Resume Interrupted Collection

If the script is interrupted, simply run it again. It will automatically resume from unfinished queries stored in the `query_state` table.

### Reset Query State

To restart query processing from scratch (without deleting collected data):

```bash
python justgiving_search.py --reset-state
```

## Configuration Parameters

- `PAGE_SIZE`: Number of results per API call (default: 100)
- `MAX_PAGES`: Maximum pages to process per query (default: 500)
- `MAX_PAGES_THRESHOLD`: Threshold for query refinement (default: 300)
- `MAX_RETRIES`: Maximum retry attempts for failed requests (default: 5)
- `INITIAL_BACKOFF`: Initial backoff time in seconds (default: 2.0)
- `MAX_RECURSION_DEPTH`: Maximum depth for query refinement (default: 3)

## Query Refinement Strategy

The script implements a recursive refinement strategy:

1. **Depth 1**: Single letters (a-z) → 26 queries
2. **Depth 2**: If a query returns ≥300 pages, refine to letter pairs (aa-zz) → 26 sub-queries
3. **Depth 3**: If a sub-query still returns ≥300 pages, refine to triplets (aaa-zzz) → 26 sub-sub-queries

Example:
```
Query "c" returns 400 pages → Refine to:
- ca, cb, cc, cd, ce, cf, cg, ch, ci, cj, ck, cl, cm, cn, co, cp, cq, cr, cs, ct, cu, cv, cw, cx, cy, cz
```

## Error Handling

The script handles various HTTP status codes:

- **200**: Success - proceed with data extraction
- **429, 503**: Rate limit or temporary server issue - retry with exponential backoff
- **401, 403**: Authentication error - non-recoverable, stops execution
- **404**: Page not found - skip and continue
- **4xx**: Client errors - non-recoverable
- **5xx**: Server errors - retry with exponential backoff

## Data Output

Each campaign is stored in the `crowdfunding` table with:
- `short_name`: Unique identifier extracted from the campaign URL
- `details`: Complete campaign data as JSONB (includes story text, metadata, etc.)

## Requirements

Install required packages:

```bash
pip install requests psycopg2-binary
```

Or use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Notes

- The script uses `ON CONFLICT DO NOTHING` to prevent duplicate entries
- Query state is tracked separately from campaign data, allowing safe resumption
- The script respects API rate limits through exponential backoff
- Large queries are automatically refined to ensure comprehensive coverage

## Citation

If you use this script, please cite:

[Your paper citation will be added here]

## License

[License information to be added]

## Contact

For questions about the data collection methodology, please contact [contact information to be added]

