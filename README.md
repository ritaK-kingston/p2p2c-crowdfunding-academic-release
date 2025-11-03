# Academic Release: Crowdfunding Motivation Classification

This repository contains the complete methodology and scripts for reproducing the crowdfunding motivation classification study. All scripts have been cleaned to remove credential references and can be run with publicly available data.

## Repository Structure

This repository is organized into two main folders:

### 📁 `search/`
Contains the data collection scripts for gathering crowdfunding data from the JustGiving API.

**Contents:**
- `justgiving_search.py` - Main data collection script
- `README.md` - Documentation for data collection methodology
- `requirements.txt` - Python dependencies

**See:** [`search/README.md`](search/README.md) for detailed usage instructions.

### 📁 `analysis/`
Contains all analysis scripts for the BERT ensemble methodology and validation procedures.

**Contents:**
- `bert_hybrid_analysis.py` - Main BERT ensemble analysis script
- `comprehensive_statistical_analysis.py` - Statistical validation scripts
- `subsample_consistency_analysis.py` - Subsample consistency validation
- `purpose_type_alignment_analysis.py` - Purpose-type alignment validation
- `keyword_robustness_analysis.py` - Keyword component robustness analysis
- `keyword_robustness_with_ensemble_context.py` - Combined keyword/ensemble reliability analysis
- `README.md` - Detailed documentation for all analysis scripts

**See:** [`analysis/README.md`](analysis/README.md) for detailed usage instructions.

## Quick Start

### 1. Data Collection

First, collect crowdfunding data using the scripts in the `search/` folder:

```bash
cd search
python justgiving_search.py
```

This will collect campaign data from the JustGiving API and store it in a PostgreSQL database.

### 2. Data Analysis

Once you have collected data, run the BERT ensemble analysis:

```bash
cd analysis
python bert_hybrid_analysis.py your_data.csv
```

This will generate classification results that can then be used with the validation scripts.

### 3. Validation

Run validation scripts to verify the methodology:

```bash
# Subsample consistency
python subsample_consistency_analysis.py bert_results.csv

# Purpose-type alignment
python purpose_type_alignment_analysis.py bert_results.csv

# Keyword robustness
python keyword_robustness_analysis.py bert_results.csv
```

## Requirements

### Search Scripts
```bash
cd search
pip install -r requirements.txt
```

### Analysis Scripts
```bash
cd analysis
pip install pandas numpy scipy matplotlib seaborn scikit-learn transformers torch
```

## Data Format

### Input for Analysis Scripts
CSV files should contain at minimum:
- An identifier column (e.g., `shortname` or `short_name`)
- A text column containing the crowdfunding narrative (e.g., `story`)

### Output from Analysis Scripts
The BERT ensemble analysis generates CSV files with:
- Original columns from input
- `clean_story`: Preprocessed text
- `primary_category`: Primary motivation category
- `all_categories`: List of all assigned categories
- `all_category_scores`: Dictionary with scores for all 9 categories
- Additional diagnostic columns

## Methodology Overview

The methodology consists of two main phases:

### Phase 1: Data Collection
- Multi-level search strategy (breadth-first with adaptive refinement)
- Fault-tolerant HTTP handling with exponential backoff
- State management for resume capability
- PostgreSQL storage with JSONB format

### Phase 2: BERT Ensemble Analysis
- **Zero-shot Classification (50% weight)**: BART-Large-MNLI for semantic understanding
- **Keyword Matching (30% weight)**: Domain-specific phrase detection
- **Emotion Analysis (15% weight)**: DistilRoBERTa emotion classification
- **Sentiment Analysis (5% weight)**: Twitter-RoBERTa sentiment analysis
- **Confidence Squared Weighting**: Applied to create proportional motivation profiles

## Citation

If you use these scripts, please cite:

[Your paper citation will be added here]

## License

[License information to be added]

## Contact

For questions about the methodology or scripts, please contact [contact information to be added]
