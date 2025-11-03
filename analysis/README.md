# Academic Release: BERT Ensemble Analysis for Crowdfunding Motivation Classification

This repository contains the analysis scripts for reproducing the BERT ensemble methodology used in the crowdfunding motivation classification study. All scripts have been cleaned to remove any credential references and can be run with publicly available data.

## Scripts Included

### 1. `bert_hybrid_analysis.py`
Main script implementing the Hybrid BERT Ensemble methodology for crowdfunding motivation classification. This script:
- Implements a multi-component ensemble (Zero-shot classification, Keyword matching, Emotion analysis, Sentiment analysis)
- Applies confidence squared weighting methodology
- Performs text preprocessing and deduplication
- Generates multi-label classification results

**Input**: CSV file with columns `shortname` (or `short_name`) and `story` containing the crowdfunding narrative text.

**Output**: CSV file with classification results including category scores, confidence values, and multi-label assignments.

**Usage**:
```python
from bert_hybrid_analysis import run_hybrid_bert_analysis_from_csv

# Run analysis on your CSV file
results_df = run_hybrid_bert_analysis_from_csv(
    input_csv='your_data.csv',
    story_column='story',
    id_column='shortname',
    sample_size=None  # Set to None for full dataset, or specify number
)
```

### 2. `comprehensive_statistical_analysis.py`
Script for comprehensive statistical validation of weighting methods. This script:
- Compares different weighting approaches (threshold-only, soft weighting, confidence squared)
- Calculates multiple statistical measures (Gini coefficient, entropy, CV, HHI, etc.)
- Generates publication-ready statistical comparisons

**Input**: CSV file with BERT analysis results (output from `bert_hybrid_analysis.py`)

**Usage**:
```python
from comprehensive_statistical_analysis import ComprehensiveStatisticalAnalyzer

analyzer = ComprehensiveStatisticalAnalyzer('bert_results.csv')
results = analyzer.run_full_analysis()
```

### 3. `subsample_consistency_analysis.py`
Script for validating robustness through subsample consistency checks. This script:
- Performs train/test split validation
- Analyzes consistency across random subsamples
- Generates visualizations of consistency metrics

**Input**: CSV file with BERT analysis results (output from `bert_hybrid_analysis.py`)

**Usage**:
```python
from subsample_consistency_analysis import SubsampleConsistencyAnalyzer

analyzer = SubsampleConsistencyAnalyzer('bert_results.csv')
train_test_results, subsample_results = analyzer.run_full_analysis()
```

### 4. `purpose_type_alignment_analysis.py`
Script for validating construct validity through purpose-type alignment. This script:
- Tests alignment between motivation scores and campaign purpose categories
- Performs ANOVA and post-hoc statistical tests
- Generates visualizations of alignment patterns

**Input**: 
- BERT results CSV (output from `bert_hybrid_analysis.py`)
- Optional: Separate CSV with campaign purpose data (if not included in BERT results)

**Usage**:
```python
from purpose_type_alignment_analysis import main

# If activity_type column is already in BERT results CSV
results_df = main('bert_results.csv')

# If using separate purpose CSV
results_df = main('bert_results.csv', purpose_csv='campaign_purposes.csv')
```

**Command Line**:
```bash
# Option 1: Activity type already in BERT results CSV
python purpose_type_alignment_analysis.py bert_results.csv

# Option 2: Separate purpose CSV
python purpose_type_alignment_analysis.py bert_results.csv campaign_purposes.csv
```

### 5. `keyword_robustness_analysis.py`
Script for validating keyword component robustness. This script:
- Analyzes whether keyword dictionaries are dominated by single keywords
- Calculates share of matches from most frequent keyword
- Computes Gini coefficients to measure keyword usage inequality
- Generates visualizations of robustness metrics

**Input**: CSV file with BERT analysis results (output from `bert_hybrid_analysis.py`, must include `clean_story` column)

**Usage**:
```python
from keyword_robustness_analysis import main

results_df, category_keyword_totals = main('bert_results.csv')
```

**Command Line**:
```bash
python keyword_robustness_analysis.py bert_results.csv
```

### 6. `keyword_robustness_with_ensemble_context.py`
Script for combining keyword robustness analysis with ensemble reliability assessment. This script:
- Combines keyword robustness metrics with zero-shot performance data
- Creates visualizations contrasting keyword weaknesses with ensemble reliability
- Demonstrates that ensemble remains reliable despite keyword issues

**Input**: 
- Keyword robustness results CSV (output from `keyword_robustness_analysis.py`)
- Ensemble component summary CSV (output from `comprehensive_statistical_analysis.py` or `ensemble_reliability_analysis.py`)
- Zero-shot correlation CSV (output from `ensemble_reliability_analysis.py`)

**Usage**:
```python
from keyword_robustness_with_ensemble_context import main

merged_df = main('keyword_robustness.csv', 'ensemble_summary.csv', 'zero_shot_correlation.csv')
```

**Command Line**:
```bash
python keyword_robustness_with_ensemble_context.py keyword_robustness.csv ensemble_summary.csv zero_shot_correlation.csv
```

## Requirements

Install required packages:
```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn transformers torch
```

## Data Format

### Input CSV for `bert_hybrid_analysis.py`
Your input CSV should contain at minimum:
- An identifier column (e.g., `shortname` or `short_name`)
- A text column containing the crowdfunding narrative (e.g., `story`)

Example:
```csv
shortname,story
story1,"This is a crowdfunding story about..."
story2,"Another story about..."
```

### Output CSV from `bert_hybrid_analysis.py`
The output CSV contains:
- Original columns from input
- `clean_story`: Preprocessed text
- `story_length`: Length in words
- `primary_category`: Primary motivation category
- `primary_confidence`: Confidence score for primary category
- `all_categories`: List of all assigned categories
- `num_categories`: Number of categories assigned
- `all_category_scores`: Dictionary with scores for all 9 categories
- Additional diagnostic columns (emotions, sentiment, etc.)

## Methodology

The BERT ensemble approach combines four analysis components:
1. **Zero-shot Classification (50% weight)**: Uses BART-Large-MNLI for semantic understanding
2. **Hybrid Keyword Matching (30% weight)**: Domain-specific phrase detection with specificity weights
3. **Emotion Analysis (15% weight)**: DistilRoBERTa emotion classification
4. **Sentiment Analysis (5% weight)**: Twitter-RoBERTa sentiment analysis

The final scores use **confidence squared weighting** where:
- Predictions ≥ threshold (0.55): weight = prob²
- Predictions < threshold: weight = (prob²) × 0.3

Results are normalized to create proportional motivation profiles for each story.

## Citation

If you use these scripts, please cite:

[Your paper citation will be added here]

## License

[License information to be added]

## Contact

For questions about the methodology or scripts, please contact [contact information to be added]

