# Academic Release: BERT Ensemble Analysis for Crowdfunding Motivation Classification

This repository contains the complete analysis pipeline for reproducing the BERT ensemble methodology used in the crowdfunding motivation classification study. All scripts have been carefully cleaned to remove any credential references and can be run with publicly available data in CSV format.

## Overview

The analysis pipeline implements a sophisticated multi-component ensemble approach that combines state-of-the-art natural language processing techniques to classify motivational content in crowdfunding narratives. The methodology addresses the challenge of multi-label classification for nuanced psychological constructs by leveraging:

1. **Semantic Understanding**: Zero-shot classification using large language models
2. **Domain-Specific Knowledge**: Curated keyword dictionaries with specificity weighting
3. **Emotional Context**: Emotion classification to capture affective signals
4. **Sentiment Analysis**: Overall sentiment signals to complement other components
5. **Confidence-Weighted Scoring**: A novel confidence squared weighting approach that preserves information while maintaining meaningful distinctions

The pipeline processes narratives through preprocessing, multi-component classification, confidence weighting, normalization, and comprehensive validation procedures to ensure reliability, construct validity, and robustness.

## Scripts Included

### 1. `bert_hybrid_analysis.py`
**Main BERT Ensemble Classification Script**

This is the core script that implements the complete Hybrid BERT Ensemble methodology for crowdfunding motivation classification. It orchestrates the entire analysis pipeline from text preprocessing through multi-component ensemble classification to final output generation.

**Key Functionality:**
- **Text Preprocessing**: Removes HTML tags, URLs, and special characters; normalizes whitespace; truncates to 2,500 characters to fit BERT's context window; performs deduplication based on text similarity
- **Four-Component Ensemble**:
  - Zero-shot Classification (50% weight): BART-Large-MNLI for semantic understanding without domain-specific training
  - Hybrid Keyword Matching (30% weight): Domain-specific phrase detection with specificity-based weighting and occurrence capping (max 3 per keyword per story)
  - Emotion Analysis (15% weight): DistilRoBERTa emotion classifier capturing emotional context (joy, sadness, anger, fear, surprise, etc.)
  - Sentiment Analysis (5% weight): Twitter-RoBERTa sentiment analysis for overall positive/negative sentiment signals
- **Confidence Squared Weighting**: Applies confidence squared methodology where predictions ≥ threshold (0.55) receive weight = prob², and predictions < threshold receive weight = (prob²) × 0.3
- **Normalization**: Normalizes all scores to sum to 1.0 for each story, creating proportional motivation profiles
- **Multi-Label Classification**: Assigns multiple categories to each narrative, reflecting the nuanced nature of motivational content

**Input**: CSV file with columns `shortname` (or `short_name`) and `story` containing the crowdfunding narrative text.

**Output**: CSV file with classification results including:
- Original input columns preserved
- `clean_story`: Preprocessed text
- `story_length`: Length in words
- `primary_category`: Primary motivation category (highest confidence)
- `primary_confidence`: Confidence score for primary category
- `all_categories`: List of all assigned categories (multi-label)
- `num_categories`: Number of categories assigned
- `all_category_scores`: Dictionary/JSON string with normalized scores for all 9 categories
- Component-level scores for diagnostic purposes

**Usage (Command Line)**:
```bash
python bert_hybrid_analysis.py your_data.csv
```

**Usage (Python Import)**:
```python
from bert_hybrid_analysis import run_hybrid_bert_analysis_from_csv

# Run analysis on your CSV file
results_df = run_hybrid_bert_analysis_from_csv(
    input_csv='your_data.csv',
    story_column='story',
    id_column='shortname',
    sample_size=None  # Set to None for full dataset, or specify number for testing
)

# Save results
results_df.to_csv('bert_results.csv', index=False)
```

### 2. `comprehensive_statistical_analysis.py`
**Statistical Validation and Method Comparison Script**

This script performs comprehensive statistical validation by comparing different weighting approaches and calculating multiple statistical measures used for academic publication. It enables researchers to evaluate the quality of the confidence squared weighting methodology compared to alternative approaches.

**Key Functionality:**
- Compares different weighting methods:
  - Threshold-only (hard thresholding)
  - Soft weighting (linear weighting)
  - Confidence squared (recommended approach)
  - Sigmoid weighting
  - And additional alternative methods
- Calculates comprehensive statistical measures:
  - **Gini Coefficient**: Measures inequality in distribution (lower is better for balance)
  - **Shannon Entropy**: Measures diversity/unpredictability (higher is better)
  - **Coefficient of Variation (CV)**: Measures relative variability
  - **Herfindahl-Hirschman Index (HHI)**: Measures market concentration/distribution concentration
  - **Skewness/Kurtosis**: Measures shape of distribution
  - **Coverage**: Percentage of stories with at least one category above threshold
  - **Stability**: Consistency across different weighting methods
- Generates publication-ready statistical comparisons and visualizations
- Provides composite "Quality Score" combining multiple metrics

**Input**: CSV file with BERT analysis results (output from `bert_hybrid_analysis.py`), specifically requiring the `all_category_scores` column.

**Usage**:
```bash
python comprehensive_statistical_analysis.py bert_results.csv
```

**Output**: 
- Statistical comparison tables
- Publication-ready visualizations
- CSV files with detailed statistical measures

### 3. `subsample_consistency_analysis.py`
**Robustness Validation Through Subsample Consistency**

This script validates the robustness of the BERT ensemble methodology by assessing consistency across different subsets of data. It implements two key validation procedures: train/test split consistency and random subsample consistency.

**Key Functionality:**
- **Train/Test Split Analysis**: Randomly splits data into 80% training and 20% testing subsets, runs the complete BERT ensemble analysis on each subset independently, and compares category distributions
- **Random Subsample Analysis**: Creates 5 independent random 50% subsamples, runs the complete analysis on each, and assesses consistency across subsamples
- Calculates correlation coefficients (Pearson's r) between distributions
- Calculates mean absolute differences in category percentages
- Calculates standard deviations across subsamples
- Generates comprehensive visualizations showing consistency metrics

**Validation Metrics:**
- **Correlation (r)**: Measures linear relationship between distributions (target: r > 0.99 for excellent consistency)
- **Mean Absolute Difference**: Average absolute difference in category percentages (target: < 0.5 percentage points)
- **Standard Deviation**: Variability across subsamples (target: < 0.2 percentage points per category)

**Input**: CSV file with BERT analysis results (output from `bert_hybrid_analysis.py`).

**Usage**:
```bash
python subsample_consistency_analysis.py bert_results.csv
```

**Output**:
- Statistical summary of consistency metrics
- Visualizations comparing train/test distributions
- Visualizations showing subsample distributions
- CSV files with detailed consistency statistics

### 4. `purpose_type_alignment_analysis.py`
**Construct Validity Validation Through Purpose-Type Alignment**

This script validates construct validity by testing whether the derived motivation scores align with known campaign "purpose" categories (the predefined types of fundraising events in the data, such as InMemory memorials, birthdays, athletic challenges, etc.). If the BERT ensemble methodology is capturing real motivational signals, certain motivations should be significantly higher in certain types of campaigns.

**Key Functionality:**
- Extracts campaign activity types from input data
- Groups narratives by activity type
- Calculates mean motivation scores for each category by activity type
- Performs one-way ANOVA to test for significant effects of campaign type on motivation scores
- Performs post-hoc Bonferroni tests to identify specific pairwise differences
- Calculates effect sizes (partial η² for ANOVA, Cohen's d for pairwise comparisons)
- Generates visualizations showing motivation scores by campaign type
- Creates heatmaps showing all motivations across top activity types

**Expected Patterns (Construct Validity):**
- **InMemory campaigns** should show significantly higher "Close to the Heart" scores
- **Challenge/athletic campaigns** should show significantly higher "Seeking Experiences" scores
- These patterns validate that the methodology captures real motivational signals

**Input**: 
- BERT results CSV (output from `bert_hybrid_analysis.py`)
- Optional: Separate CSV with campaign purpose/activity type data (if not included in BERT results)

**Usage (Activity type in BERT results)**:
```bash
python purpose_type_alignment_analysis.py bert_results.csv
```

**Usage (Separate purpose CSV)**:
```bash
python purpose_type_alignment_analysis.py bert_results.csv campaign_purposes.csv
```

**Output**:
- ANOVA results and post-hoc test statistics
- Effect size calculations
- Visualizations showing motivation scores by campaign type
- Heatmaps comparing all motivations across activity types

### 5. `keyword_robustness_analysis.py`
**Keyword Component Robustness Assessment**

This script validates whether each motivation category's keyword dictionary is dominated by only one or two keywords. Over-reliance on single keywords would indicate fragility in the keyword component.

**Key Functionality:**
- Re-matches keyword dictionaries against preprocessed narrative text (`clean_story`)
- For each category, counts occurrences of each keyword
- Calculates the share of matches attributable to the most frequent keyword
- Calculates Gini coefficients to measure inequality of keyword usage
- Generates visualizations showing robustness metrics per category
- Exports detailed CSV with keyword match statistics

**Robustness Metrics:**
- **Most Frequent Keyword Share**: Percentage of all keyword matches that come from the single most frequent keyword (target: < 30% for good robustness)
- **Gini Coefficient**: Measures inequality of keyword usage (target: < 0.5 for relatively even distribution)

**Interpretation:**
- Low share and low Gini indicate robust keyword dictionaries that capture broad language patterns
- High share and high Gini indicate potential over-reliance on specific terms

**Input**: CSV file with BERT analysis results (output from `bert_hybrid_analysis.py`), must include `clean_story` column for re-matching keywords.

**Usage**:
```bash
python keyword_robustness_analysis.py bert_results.csv
```

**Output**:
- Robustness metrics per category (most frequent share, Gini coefficient)
- Visualizations showing robustness across categories
- CSV file with detailed keyword match statistics

### 6. `keyword_robustness_with_ensemble_context.py`
**Combined Keyword Robustness and Ensemble Reliability Analysis**

This script combines keyword robustness analysis with ensemble reliability assessment to provide a comprehensive view of component-level issues within the context of overall ensemble performance. It demonstrates that despite potential keyword weaknesses, the ensemble remains reliable due to the dominance and strength of the zero-shot component.

**Key Functionality:**
- Loads keyword robustness results (from `keyword_robustness_analysis.py`)
- Loads ensemble component contribution data (zero-shot vs. keywords)
- Loads zero-shot correlation data (how well zero-shot explains final scores)
- Creates visualizations contrasting keyword weaknesses with ensemble reliability
- Demonstrates that ensemble design provides redundancy that mitigates individual component weaknesses

**Key Insight:**
Even if keyword dictionaries show concentration issues, the ensemble remains highly reliable because:
1. Zero-shot component carries 50% weight and explains high variance in final scores
2. Keywords contribute relatively small signal compared to zero-shot
3. Multi-component ensemble provides redundancy

**Input**: 
- Keyword robustness results CSV (output from `keyword_robustness_analysis.py`)
- Ensemble component summary CSV (from ensemble reliability analysis or comprehensive statistical analysis)
- Zero-shot correlation CSV (from ensemble reliability analysis)

**Usage**:
```bash
python keyword_robustness_with_ensemble_context.py keyword_robustness.csv ensemble_summary.csv zero_shot_correlation.csv
```

**Output**:
- Comprehensive visualization showing keyword robustness vs. ensemble reliability
- Summary statistics demonstrating ensemble reliability despite keyword issues
- Visualizations of component contributions and zero-shot R² values

## Requirements

Install required packages:

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn transformers torch
```

### Core Dependencies
- **`pandas`**, **`numpy`**: Data manipulation and numerical computing
- **`scipy`**: Statistical functions (entropy, chi-square tests, ANOVA, etc.)
- **`matplotlib`**, **`seaborn`**: Publication-ready visualization
- **`scikit-learn`**: Machine learning utilities (train/test splits, metrics, etc.)

### Deep Learning Dependencies
- **`transformers`**: Hugging Face transformers library for BERT models
  - BART-Large-MNLI (zero-shot classification)
  - DistilRoBERTa (emotion analysis)
  - Twitter-RoBERTa (sentiment analysis)
- **`torch`**: PyTorch backend for transformers

**Note**: Running the analysis scripts requires substantial computational resources. The BERT models are large (BART-Large-MNLI is ~1.6GB) and may require:
- GPU acceleration for reasonable processing times on large datasets
- Substantial RAM (8GB+ recommended for large datasets)
- Significant processing time (hours for datasets with 10,000+ narratives)

## Data Format

### Input CSV for `bert_hybrid_analysis.py`

Your input CSV should contain at minimum:
- An identifier column (e.g., `shortname` or `short_name`) - unique identifier for each narrative
- A text column containing the crowdfunding narrative (e.g., `story`) - the full narrative text to be analyzed

Optional columns that enhance analysis:
- `activity_type` - Campaign type (e.g., "InMemory", "Birthday", "Challenge") - used in purpose-type alignment validation
- Any other metadata you wish to preserve in the output

**Example Input CSV:**
```csv
shortname,story,activity_type
story1,"This is a crowdfunding story about raising money for a memorial...","InMemory"
story2,"Another story about a personal marathon challenge...","Challenge"
story3,"A story about helping others in need...","Other"
```

### Output CSV from `bert_hybrid_analysis.py`

The output CSV contains all original columns plus the following analysis results:

- **`clean_story`**: Preprocessed version of the narrative text (HTML/URLs removed, whitespace normalized)
- **`story_length`**: Length of the narrative in words (after preprocessing)
- **`primary_category`**: The motivation category with the highest confidence score
- **`primary_confidence`**: Confidence score for the primary category (before normalization)
- **`all_categories`**: List of all categories assigned to this narrative (multi-label classification, comma-separated or JSON list)
- **`num_categories`**: Number of categories assigned to this narrative
- **`all_category_scores`**: Dictionary/JSON string with normalized scores for all 9 categories:
  - `Close to the Heart`
  - `Close to Home`
  - `Altruism and Empathy`
  - `Moral Obligation`
  - `Social Standing`
  - `Personal Development`
  - `Seeking Experiences`
  - `Stewardship`
  - `Advocacy`
- **Component scores**: Individual scores from zero-shot, keywords, emotion, and sentiment components (for diagnostic purposes)

**Example Output (simplified):**
```csv
shortname,story,primary_category,all_categories,num_categories,all_category_scores
story1,"Original story...","Close to the Heart","Close to the Heart, Altruism and Empathy",2,"{'Close to the Heart': 0.45, 'Altruism and Empathy': 0.32, ...}"
```

## Methodology

### BERT Ensemble Architecture

The BERT ensemble approach combines four complementary analysis components:

#### 1. Zero-shot Classification (50% weight)
- **Model**: BART-Large-MNLI (BartForSequenceClassification)
- **Purpose**: Semantic understanding of narrative content without requiring domain-specific training data
- **How it works**: Uses natural language inference capabilities to classify narratives based on semantic content
- **Strength**: Captures nuanced semantic meaning and generalizes well to new domains

#### 2. Hybrid Keyword Matching (30% weight)
- **Approach**: Domain-specific phrase detection using curated keyword dictionaries
- **Purpose**: Captures domain-specific language patterns that may not be evident in general language models
- **How it works**:
  - Each motivation category has a curated dictionary of keywords/phrases
  - Keywords are weighted by specificity (more specific terms get higher weights)
  - Occurrences are capped at 3 per keyword per story to prevent dominance
  - Matches are summed and normalized per category
- **Strength**: Captures domain-specific terminology and explicit motivational language

#### 3. Emotion Analysis (15% weight)
- **Model**: DistilRoBERTa emotion classifier
- **Purpose**: Captures emotional context that may indicate underlying motivations
- **How it works**: Classifies text into emotion categories (joy, sadness, anger, fear, surprise, etc.)
- **Emotion-to-Motivation Mapping**: Specific emotions are mapped to motivation categories (e.g., sadness → Close to the Heart, joy → Seeking Experiences)
- **Strength**: Captures affective signals that complement semantic and keyword-based analysis

#### 4. Sentiment Analysis (5% weight)
- **Model**: Twitter-RoBERTa sentiment classifier
- **Purpose**: Provides overall positive/negative sentiment signals
- **How it works**: Classifies text as positive, negative, or neutral sentiment
- **Strength**: Captures overall tone that complements other components

### Confidence Squared Weighting

The ensemble scores are combined using a weighted average, then confidence squared weighting is applied:

**For predictions ≥ threshold (0.55)**:
```
weight = prob²
```

**For predictions < threshold**:
```
weight = (prob²) × 0.3
```

**Rationale:**
- Squaring amplifies high-confidence predictions while reducing low-confidence noise
- Below-threshold predictions are preserved (not discarded) but receive reduced influence (30% multiplier)
- This balances maintaining meaningful distinctions between categories while avoiding information loss from hard thresholding

### Normalization

All scores are normalized to sum to 1.0 for each story:
```
normalized_score(category) = weighted_score(category) / sum(all_weighted_scores)
```

This creates proportional motivation profiles showing relative emphasis across categories, enabling meaningful comparison across narratives regardless of overall score magnitude.

### Multi-Label Classification

The methodology assigns multiple categories to each narrative, reflecting the nuanced nature of motivational content. A threshold of 0.55 (determined through optimization balancing coverage, entropy, and category balance) is used to determine category assignments, but confidence squared weighting preserves information from all predictions.

**Results**: 92.6% of narratives are classified into multiple categories, demonstrating the methodology's ability to capture nuanced motivational content.

## Validation Procedures

The repository includes comprehensive validation procedures to ensure methodological rigor:

### 1. Subsample Consistency
Validates robustness by checking consistency across different data subsets (train/test splits, random subsamples). Excellent consistency (correlations > 0.999) indicates reliable and reproducible results.

### 2. Purpose-Type Alignment
Validates construct validity by checking alignment between motivation scores and known campaign types. Significant alignments (e.g., InMemory campaigns → Close to the Heart) confirm that the methodology captures real motivational signals.

### 3. Keyword Robustness
Assesses whether keyword dictionaries are overly reliant on single terms. This identifies potential weaknesses in the keyword component, though ensemble reliability analysis demonstrates that the overall methodology remains robust.

### 4. Cross-Model Agreement
Evaluates complementarity between ensemble components. Low correlations between components are desirable as they indicate complementary rather than redundant signals.

## Use Cases

This repository is suitable for:

- **Researchers**: Reproducing and extending research on computational analysis of motivational content in crowdfunding or similar narrative domains
- **Practitioners**: Implementing automated classification of motivational narratives for content analysis, recommendation systems, or research applications
- **Students**: Learning about ensemble NLP methodologies, multi-label classification, confidence weighting, and validation procedures
- **Organizations**: Analyzing motivational patterns in charitable campaigns or similar narrative datasets

## Notes

- All scripts have been cleaned to remove credential references (database connections, API keys, etc.)
- Scripts are designed to work with CSV input files for transparency and reproducibility
- The methodology has been validated through multiple procedures demonstrating reliability, construct validity, and robustness
- Processing large datasets requires substantial computational resources (GPU recommended)
- The confidence squared weighting methodology represents a novel contribution to the field

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
