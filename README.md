# Crowdfunding Motivation Classification

This repository contains the complete, reproducible methodology and scripts for analyzing motivational content in crowdfunding narratives using a BERT-based ensemble approach. All scripts have been carefully cleaned to remove credential references and can be run with publicly available data. This repository provides everything needed to replicate the data collection, preprocessing, classification, and validation procedures described in the associated research.

## Repository Purpose

The primary goal of this repository is to provide transparency and reproducibility for research on automated classification of motivational content in crowdfunding narratives. The methodology combines multiple natural language processing approaches to classify fundraiser narratives into nine distinct motivation categories (Close to the Heart, Close to Home, Altruism and Empathy, Moral Obligation, Social Standing, Personal Development, Seeking Experiences, Stewardship, and Advocacy) using a multi-component BERT ensemble.

This work addresses several key challenges in computational social science:
- **Multi-label classification** at scale for nuanced psychological constructs
- **Ensemble methodology** that combines semantic understanding, domain-specific keyword matching, emotion analysis, and sentiment analysis
- **Validation procedures** that ensure reliability, construct validity, and robustness
- **Confidence-weighted scoring** that preserves information from below-threshold predictions while maintaining meaningful distinctions

## Repository Structure

This repository is organized into two main components, each serving a distinct phase of the research pipeline:

### 📁 `search/`
Contains the data collection infrastructure for gathering crowdfunding campaign data from the JustGiving platform. This component implements a sophisticated, fault-tolerant web scraping and API interaction system that handles rate limiting, error recovery, and state management to enable large-scale data collection.

**Key Features:**
- Multi-level recursive search strategy for comprehensive coverage
- Exponential backoff retry logic for robust error handling
- State tracking for resumable collection sessions
- PostgreSQL integration for structured data storage

**Contents:**
- `justgiving_search.py` - Main data collection script implementing recursive query refinement
- `README.md` - Comprehensive documentation for the data collection methodology
- `requirements.txt` - Python dependencies for data collection

**See:** [`search/README.md`](search/README.md) for detailed usage instructions, configuration, and technical specifications.

### 📁 `analysis/`
Contains the complete analytical pipeline for classifying crowdfunding narratives using a BERT-based ensemble methodology. This component implements state-of-the-art natural language processing techniques to extract and classify motivational content, along with comprehensive validation procedures to ensure methodological rigor.

**Key Features:**
- Multi-component ensemble combining zero-shot classification, keyword matching, emotion analysis, and sentiment analysis
- Confidence squared weighting methodology for creating proportional motivation profiles
- Comprehensive validation scripts for subsample consistency, construct validity, and component robustness
- Statistical comparison tools for evaluating different weighting approaches

**Contents:**
- `bert_hybrid_analysis.py` - Main BERT ensemble classification script
- `comprehensive_statistical_analysis.py` - Statistical validation and method comparison
- `subsample_consistency_analysis.py` - Robustness validation through subsample analysis
- `purpose_type_alignment_analysis.py` - Construct validity validation through purpose-type alignment
- `keyword_robustness_analysis.py` - Keyword component robustness assessment
- `keyword_robustness_with_ensemble_context.py` - Combined keyword/ensemble reliability analysis
- `README.md` - Detailed documentation for all analysis scripts and methodology

**See:** [`analysis/README.md`](analysis/README.md) for detailed usage instructions, methodology overview, and technical specifications.

## How It Works

### Data Collection Workflow (`search/`)

The data collection system uses a sophisticated recursive search strategy:

1. **Initial Query Generation**: Starts with single-letter queries (a-z) to maximize breadth of coverage
2. **API Interaction**: Fetches campaign data from JustGiving API with pagination support
3. **Adaptive Refinement**: When queries return too many results (≥300 pages), automatically refines into sub-queries (e.g., "c" → "ca", "cb", ... "cz")
4. **State Management**: Tracks query progress in database to enable resuming interrupted sessions
5. **Fault Tolerance**: Implements exponential backoff for rate limits and server errors
6. **Data Storage**: Stores complete campaign data as JSONB in PostgreSQL for flexible querying

This approach ensures comprehensive coverage of the JustGiving platform while handling API limitations and network issues gracefully.

### Analysis Workflow (`analysis/`)

The analysis pipeline processes collected narratives through multiple stages:

1. **Text Preprocessing**: 
   - Removes HTML tags, URLs, and special characters
   - Normalizes whitespace
   - Truncates to 2,500 characters to fit within BERT's context window
   - Deduplicates based on text similarity

2. **Multi-Component Ensemble Classification**:
   - **Zero-shot Classification (50% weight)**: Uses BART-Large-MNLI for semantic understanding without domain-specific training
   - **Keyword Matching (30% weight)**: Domain-specific phrase detection with specificity-based weighting and occurrence capping (max 3 per keyword)
   - **Emotion Analysis (15% weight)**: DistilRoBERTa emotion classifier capturing emotional context
   - **Sentiment Analysis (5% weight)**: Twitter-RoBERTa sentiment analysis for overall sentiment signals

3. **Confidence Squared Weighting**:
   - Predictions ≥ threshold (0.55): weight = prob²
   - Predictions < threshold: weight = (prob²) × 0.3
   - Preserves information from below-threshold predictions while maintaining distinctions

4. **Normalization**: 
   - Scores are normalized to sum to 1.0 for each story
   - Creates proportional motivation profiles showing relative emphasis across categories

5. **Validation**: 
   - Multiple validation procedures ensure reliability, construct validity, and robustness
   - Subsample consistency checks, purpose-type alignment, keyword robustness analysis

## Quick Start

### 1. Data Collection

First, collect crowdfunding data using the scripts in the `search/` folder. You'll need:
- A JustGiving API App ID (register at [JustGiving API](https://developers.justgiving.com/))
- A PostgreSQL database configured with appropriate credentials

```bash
cd search
# Configure your API credentials and database connection
python justgiving_search.py
```

This will collect campaign data from the JustGiving API and store it in a PostgreSQL database. The script can be interrupted and resumed, making it suitable for large-scale collection over extended periods.

### 2. Data Extraction and Preparation

Extract the collected data from PostgreSQL into a CSV format suitable for analysis:

```sql
-- Example SQL to export data
SELECT 
    short_name,
    details->>'story' AS story,
    details->>'activityType' AS activity_type
FROM crowdfunding
WHERE details->>'story' IS NOT NULL;
```

Save the results as a CSV file with columns for identifier (e.g., `shortname` or `short_name`) and narrative text (e.g., `story`).

### 3. BERT Ensemble Analysis

Once you have prepared your data CSV, run the BERT ensemble classification:

```bash
cd analysis
python bert_hybrid_analysis.py your_data.csv
```

This will generate classification results including:
- Multi-label category assignments
- Confidence scores for each category
- Normalized motivation profiles
- Component-level scores (zero-shot, keywords, emotion, sentiment)

**Output**: CSV file with classification results that can be used for further analysis and validation.

### 4. Validation

Run validation scripts to verify the methodology's reliability and validity:

```bash
# Subsample consistency (robustness check)
python subsample_consistency_analysis.py bert_results.csv

# Purpose-type alignment (construct validity)
python purpose_type_alignment_analysis.py bert_results.csv

# Keyword robustness (component reliability)
python keyword_robustness_analysis.py bert_results.csv
```

Each validation script generates statistical reports and visualizations demonstrating the methodology's rigor.

### 5. Statistical Comparison

Compare different weighting approaches and generate publication-ready statistics:

```bash
python comprehensive_statistical_analysis.py bert_results.csv
```

This generates comprehensive statistical comparisons including Gini coefficients, entropy, coverage metrics, and other measures used for academic publication.

## Requirements

### Search Scripts (`search/`)
```bash
cd search
pip install -r requirements.txt
```

Dependencies:
- `requests` - HTTP library for API interactions
- `psycopg2-binary` - PostgreSQL database adapter

### Analysis Scripts (`analysis/`)
```bash
cd analysis
pip install pandas numpy scipy matplotlib seaborn scikit-learn transformers torch
```

Dependencies:
- `pandas`, `numpy` - Data manipulation and numerical computing
- `scipy` - Statistical functions
- `matplotlib`, `seaborn` - Visualization
- `scikit-learn` - Machine learning utilities
- `transformers`, `torch` - Hugging Face transformers and PyTorch for BERT models

**Note**: Running the analysis scripts requires substantial computational resources. The BERT models are large and may require GPU acceleration for reasonable processing times on large datasets.

## Data Format

### Input for Analysis Scripts

CSV files should contain at minimum:
- An identifier column (e.g., `shortname` or `short_name`) - unique identifier for each narrative
- A text column containing the crowdfunding narrative (e.g., `story`) - the full narrative text to be analyzed

Optional columns that enhance analysis:
- `activity_type` - Campaign type (e.g., "InMemory", "Birthday", "Challenge")
- Any other metadata you wish to preserve in the output

Example:
```csv
shortname,story,activity_type
story1,"This is a crowdfunding story about raising money for...","InMemory"
story2,"Another story about a personal challenge...","Challenge"
```

### Output from Analysis Scripts

The BERT ensemble analysis generates CSV files with:
- **Original columns**: All input columns are preserved
- **`clean_story`**: Preprocessed version of the narrative text
- **`story_length`**: Length of the narrative in words
- **`primary_category`**: The category with the highest confidence score
- **`primary_confidence`**: Confidence score for the primary category
- **`all_categories`**: List of all categories assigned (multi-label classification)
- **`num_categories`**: Number of categories assigned to this narrative
- **`all_category_scores`**: Dictionary/JSON string with normalized scores for all 9 categories
- **Component scores**: Individual scores from zero-shot, keywords, emotion, and sentiment components

This output format enables further statistical analysis, validation, and visualization.

## Methodology Overview

### Phase 1: Data Collection

The data collection phase uses a sophisticated multi-level search strategy:

1. **Breadth-First Initialization**: Starts with comprehensive single-letter queries (a-z) to maximize coverage
2. **Adaptive Query Refinement**: Automatically refines queries that return too many results (≥300 pages) into sub-queries recursively
3. **Fault-Tolerant HTTP Handling**: Implements exponential backoff retry logic for rate limits, server errors, and network issues
4. **State Management**: Tracks query progress in database (`query_state` table) to enable resuming interrupted collection sessions
5. **Resume Capability**: Can seamlessly resume from where it left off without duplicating work
6. **PostgreSQL Storage**: Stores complete campaign data as JSONB format for flexible querying and data extraction

This approach ensures comprehensive, reliable data collection while respecting API limitations and handling real-world network and server issues.

### Phase 2: BERT Ensemble Analysis

The analysis phase implements a multi-component ensemble approach:

1. **Text Preprocessing**:
   - HTML tag and URL removal
   - Whitespace normalization
   - Length filtering and truncation to 2,500 characters
   - Deduplication based on text similarity

2. **Four-Component Ensemble**:
   - **Zero-shot Classification (50% weight)**: BART-Large-MNLI provides semantic understanding without requiring domain-specific training data. This component leverages the model's natural language inference capabilities to classify narratives based on their semantic content.
   - **Hybrid Keyword Matching (30% weight)**: Domain-specific phrase detection using curated keyword dictionaries for each motivation category. Keywords are weighted by specificity and capped at 3 occurrences per story to prevent dominance.
   - **Emotion Analysis (15% weight)**: DistilRoBERTa emotion classifier captures emotional context (joy, sadness, anger, fear, etc.) that may indicate underlying motivations.
   - **Sentiment Analysis (5% weight)**: Twitter-RoBERTa sentiment analysis provides overall positive/negative sentiment signals that complement the other components.

3. **Confidence Squared Weighting**:
   - For predictions above the threshold (≥0.55): weight = prob² (amplifies high-confidence predictions)
   - For predictions below the threshold (<0.55): weight = (prob²) × 0.3 (preserves information while reducing influence)
   - This approach balances maintaining meaningful distinctions between categories while avoiding information loss from hard thresholding

4. **Normalization**:
   - All scores are normalized to sum to 1.0 for each story
   - Creates proportional motivation profiles showing relative emphasis across categories
   - Enables meaningful comparison across narratives regardless of overall score magnitude

5. **Multi-Label Classification**:
   - Each narrative can be assigned multiple categories (reflecting the nuanced nature of motivational content)
   - Threshold optimization (0.55) balances coverage, entropy, and category balance
   - Results show that 92.6% of narratives are classified into multiple categories, demonstrating the methodology's ability to capture nuanced motivational content

6. **Validation Procedures**:
   - **Subsample Consistency**: Validates robustness through train/test splits and random subsampling
   - **Purpose-Type Alignment**: Validates construct validity by checking alignment with known campaign types
   - **Keyword Robustness**: Assesses whether keyword dictionaries are overly reliant on single terms
   - **Cross-Model Agreement**: Evaluates complementarity between ensemble components

## Use Cases

This repository is suitable for:

- **Researchers**: Reproducing and extending research on computational analysis of motivational content in crowdfunding or similar domains
- **Practitioners**: Implementing automated classification of motivational narratives for content analysis, recommendation systems, or research applications
- **Students**: Learning about ensemble NLP methodologies, multi-label classification, and validation procedures
- **Organizations**: Analyzing motivational patterns in charitable campaigns or similar narrative datasets

## Contributing

This repository is maintained for academic transparency and reproducibility. If you identify bugs, have suggestions for improvements, or wish to report issues, please use the repository's issue tracking system.

## License

Copyright 2024 [Rita Kottasz, Matthew Wade, Claire van Teunenbroek]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Acknowledgments

This repository is the result of extensive research into computational methods for analyzing motivational content in crowdfunding narratives. The methodology combines state-of-the-art natural language processing techniques with rigorous validation procedures to ensure reliable and valid classification.
