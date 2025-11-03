#!/usr/bin/env python3
"""
Hybrid BERT Analysis for Crowdfunding Motivations

ACADEMIC RELEASE VERSION: This script reads from CSV files instead of databases.
No credential information is required.

Input CSV Requirements:
- Must contain a column with story text (default: 'story')
- Should contain an identifier column (default: 'shortname' or 'short_name')
- Stories should be at least 50 characters long

This script implements the Hybrid Approach (Strategy 6) with:
1. Contextual keywords replacing generic words
2. Multi-word phrases for better discrimination
3. Weighted scoring based on specificity
4. Phrase-based matching for context
5. TF-IDF discriminative terms
6. Combined scoring with appropriate weights
"""

import os
import sys
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# BERT and Transformers imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    print("✅ Transformers library loaded successfully")
except ImportError:
    print("❌ Transformers library not found. Installing...")
    os.system("pip install transformers torch")
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch

# Refined keyword mapping with Hybrid Approach
REFINED_MOTIVATION_KEYWORDS = {
    'Advocacy': [
        'raise awareness for', 'campaign to support', 'advocate for change', 'speak out against', 
        'fight for justice', 'stand up for', 'policy reform', 'equality rights', 'social justice', 
        'activism', 'protest', 'petition', 'voice concerns', 'make a difference', 'social movement'
    ],
    'Altruism and Empathy': [
        'save lives', 'caring for others', 'kindness', 'support those in need', 'change lives', 
        'bring hope', 'selfless giving', 'give hope', 'anonymous donation', 'concern for others', 
        'pure giving', 'act of kindness', 'help others', 'relieve suffering', 'compassionate giving'
    ],
    'Close to Home': [
        'local community', 'our neighborhood', 'nearby school', 'regional hospital', 'area development', 
        'community project', 'local charity', 'neighborhood initiative', 'regional support', 'nearby cause', 
        'local school', 'community initiative', 'area support', 'local development', 'community cause'
    ],
    'Stewardship': [
        'donation management', 'fund allocation', 'charity oversight', 'financial stewardship', 'resource management', 
        'ensure funds are used', 'directly support', 'transparent giving', 'responsible management', 'efficient allocation', 
        'accountable use', 'fund accountability', 'charity administration', 'transparent donation', 'financial oversight'
    ],
    'Seeking Experiences': [
        'challenge myself', 'personal challenge', 'charity event', 'fundraising challenge', 'sponsored activity', 
        'charity run', 'challenge event', 'fundraising event', 'marathon training', 'sponsored walk', 
        'exciting challenge', 'personal achievement', 'adventure', 'thrilling experience', 'participate in'
    ],
    'Moral Obligation': [
        'justice', 'compassion', 'humanity', 'help others', 'moral duty', 'ethical giving', 'compassionate', 
        'commitment to help', 'duty to support', 'moral responsibility', 'ethical obligation', 'human rights', 
        'social responsibility', 'moral imperative', 'ethical commitment'
    ],
    'Close to the Heart': [
        'in memory of', 'close to my heart', 'personal connection', 'family member', 'loved one', 
        'because of my experience', 'family friend', 'personal loss', 'memory tribute', 'dedicated to', 
        'personal experience', 'close family', 'personal story', 'family support', 'personal dedication'
    ],
    'Personal Development': [
        'personal growth', 'learn new skills', 'develop myself', 'overcome challenges', 'achieve goals', 
        'personal journey', 'skill development', 'learning journey', 'self-improvement', 'capability building', 
        'personal achievement', 'growth opportunity', 'develop skills', 'personal transformation', 'skill building'
    ],
    'Social Standing': [
        'like and share', 'friends support', 'social media', 'post about', 'brand recognition', 'support me', 
        'social recognition', 'identity expression', 'social influence', 'networking', 'social status', 
        'public image', 'social visibility', 'social networking', 'social presence'
    ]
}

# Specificity weights for Hybrid Approach
SPECIFICITY_WEIGHTS = {
    # High specificity (1.0) - Domain-specific terms
    'marathon training': 1.0, 'charity run': 1.0, 'fundraising event': 1.0, 'sponsored walk': 1.0,
    'donation management': 1.0, 'fund allocation': 1.0, 'charity oversight': 1.0, 'financial stewardship': 1.0,
    'local community': 1.0, 'our neighborhood': 1.0, 'nearby school': 1.0, 'regional hospital': 1.0,
    'raise awareness for': 1.0, 'campaign to support': 1.0, 'advocate for change': 1.0,
    'in memory of': 1.0, 'close to my heart': 1.0, 'personal connection': 1.0,
    'challenge myself': 1.0, 'personal challenge': 1.0, 'charity event': 1.0,
    'personal growth': 1.0, 'learn new skills': 1.0, 'develop myself': 1.0,
    'like and share': 1.0, 'friends support': 1.0, 'social media': 1.0,
    
    # Medium specificity (0.7) - Moderately specific terms
    'community project': 0.7, 'local charity': 0.7, 'neighborhood initiative': 0.7,
    'transparent giving': 0.7, 'responsible management': 0.7, 'efficient allocation': 0.7,
    'policy reform': 0.7, 'equality rights': 0.7, 'social justice': 0.7,
    'family member': 0.7, 'loved one': 0.7, 'personal experience': 0.7,
    'fundraising challenge': 0.7, 'sponsored activity': 0.7, 'exciting challenge': 0.7,
    'skill development': 0.7, 'learning journey': 0.7, 'self-improvement': 0.7,
    'social recognition': 0.7, 'identity expression': 0.7, 'social influence': 0.7,
    'caring for others': 0.7, 'support those in need': 0.7, 'change lives': 0.7,
    'justice': 0.7, 'compassion': 0.7, 'humanity': 0.7,
    
    # Low specificity (0.3) - Generic terms
    'help others': 0.3, 'support': 0.3, 'help': 0.3, 'community': 0.3, 'local': 0.3,
    'challenge': 0.3, 'event': 0.3, 'experience': 0.3, 'growth': 0.3, 'development': 0.3,
    'family': 0.3, 'friend': 0.3, 'personal': 0.3, 'social': 0.3, 'support': 0.3,
    
    # Generic (0.1) - Very generic terms
    'fun': 0.1, 'giving': 0.1, 'use': 0.1, 'like': 0.1, 'share': 0.1, 'post': 0.1, 'follow': 0.1
}

# Category priors to reduce bias (tuneable). Values >1 boost; <1 downweight.
CATEGORY_PRIORS = {
    'Seeking Experiences': 0.85,
    'Close to Home': 1.10,
    'Close to the Heart': 1.10,
    'Advocacy': 1.10,
    'Altruism and Empathy': 1.15,
    'Moral Obligation': 1.15,
    'Personal Development': 1.10,
    'Social Standing': 1.05,
    'Stewardship': 1.10,
}

# Calibration/config
TEMPERATURE = 0.7  # <1.0 sharpens softmax for clearer separation
ZERO_SHOT_TOP_K = 3  # keep only top-K zero-shot labels per text
MIN_KEYWORD_WEIGHT = 0.5  # ignore very generic keyword hits below this weight

# Multi-label classification settings
MULTI_LABEL_THRESHOLD = 0.55  # optimal threshold determined by systematic optimization
MAX_CATEGORIES_PER_STORY = 5  # maximum number of categories to assign per story

def preprocess_text(text):
    """Clean and preprocess text for BERT analysis."""
    if not text or pd.isna(text):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate if too long (BERT has 512 token limit, roughly 2000-3000 characters)
    if len(text) > 2500:  # Leave room for special tokens and safety margin
        text = text[:2500] + "..."
    
    return text

def _compile_keyword_patterns(motivation_keywords):
    """Pre-compile regex patterns for keywords with word boundaries and phrase support."""
    patterns = {}
    for category, keywords in motivation_keywords.items():
        compiled = []
        for kw in keywords:
            # Escape regex special chars but keep spaces; use word boundaries around words
            escaped = re.escape(kw)
            # Replace escaped spaces with flexible whitespace
            escaped = escaped.replace(r"\ ", r"\s+")
            # Word boundary only if alnum on ends
            pattern = rf"(?i)(?<!\w){escaped}(?!\w)"
            compiled.append(re.compile(pattern))
        patterns[category] = compiled
    return patterns

def load_bert_models():
    """Load BERT models for analysis."""
    print("🔄 Loading BERT models...")
    
    models = {}
    
    # 1. Emotion classifier
    try:
        models['emotion'] = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        print("✅ Emotion classifier loaded")
    except Exception as e:
        print(f"⚠️ Emotion classifier failed: {e}")
        models['emotion'] = None
    
    # 2. Sentiment classifier
    try:
        models['sentiment'] = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True
        )
        print("✅ Sentiment classifier loaded")
    except Exception as e:
        print(f"⚠️ Sentiment classifier failed: {e}")
        models['sentiment'] = None
    
    # 3. Zero-shot classifier
    try:
        models['zero_shot'] = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        print("✅ Zero-shot classifier loaded")
    except Exception as e:
        print(f"⚠️ Zero-shot classifier failed: {e}")
        models['zero_shot'] = None
    
    return models

def analyze_emotions_bert(text, emotion_classifier):
    """Analyze emotions using BERT."""
    if not text or len(text.strip()) < 10:
        return {}
    
    try:
        results = emotion_classifier(text)
        
        # Handle nested list structure
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                results = results[0]
        
        emotions = {}
        for result in results:
            if isinstance(result, dict) and 'label' in result and 'score' in result:
                emotions[result['label']] = result['score']
        
        return emotions
        
    except Exception as e:
        print(f"⚠️ Error in emotion analysis: {e}")
        return {}

def analyze_sentiment_bert(text, sentiment_classifier):
    """Analyze sentiment using BERT."""
    if not text or len(text.strip()) < 10:
        return {'sentiment': 'neutral', 'confidence': 0.0}
    
    try:
        results = sentiment_classifier(text)
        
        # Handle nested list structure
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                results = results[0]
        
        scores = {}
        for result in results:
            if isinstance(result, dict) and 'label' in result and 'score' in result:
                scores[result['label']] = result['score']
        
        if scores:
            dominant_sentiment = max(scores, key=scores.get)
            confidence = scores[dominant_sentiment]
        else:
            dominant_sentiment = 'neutral'
            confidence = 0.0
        
        return {
            'sentiment': dominant_sentiment,
            'confidence': confidence,
            'scores': scores
        }
        
    except Exception as e:
        print(f"⚠️ Error in sentiment analysis: {e}")
        return {'sentiment': 'neutral', 'confidence': 0.0}

def analyze_motivation_zero_shot(text, zero_shot_classifier):
    """Analyze motivation using zero-shot classification."""
    if not text or len(text.strip()) < 10:
        return {}
    
    try:
        # Define the 9 motivational categories as labels
        candidate_labels = list(REFINED_MOTIVATION_KEYWORDS.keys())
        
        result = zero_shot_classifier(text, candidate_labels)
        labels = result['labels']
        scores = result['scores']
        all_scores = dict(zip(labels, scores))
        # Keep only top-K labels to reduce noise; renormalize
        top_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:ZERO_SHOT_TOP_K]
        top_sum = sum(s for _, s in top_items) or 1.0
        pruned_scores = {lab: (sc / top_sum) for lab, sc in top_items}
        pred_label = top_items[0][0] if top_items else None
        pred_conf = pruned_scores.get(pred_label, 0.0) if pred_label else 0.0
        return {
            'predicted_label': pred_label,
            'confidence': pred_conf,
            'all_scores': pruned_scores
        }
        
    except Exception as e:
        print(f"⚠️ Error in zero-shot analysis: {e}")
        return {}

def calculate_hybrid_keyword_scores(text, keyword_patterns):
    """Calculate hybrid keyword scores using regex with specificity weights and frequency capping."""
    if not text:
        return {cat: 0.0 for cat in keyword_patterns.keys()}
    scores = {}
    for category, patterns in keyword_patterns.items():
        score = 0.0
        for pat in patterns:
            kw = pat.pattern
            # Retrieve original keyword string from pattern by stripping regex meta (best-effort)
            # Fallback default weight if not found
            weight = 0.5
            for k, w in SPECIFICITY_WEIGHTS.items():
                if re.sub(r"\\s\+", " ", kw.lower()).find(re.escape(k.lower())) != -1:
                    weight = w
                    break
            # Threshold weak matches
            if weight < MIN_KEYWORD_WEIGHT:
                continue
            # Count occurrences (cap at 3 to avoid long repeats dominating)
            matches = len(pat.findall(text))
            if matches:
                score += weight * min(matches, 3)
        scores[category] = score
    return scores

def map_to_motivation_category_hybrid(emotion_results, sentiment_results, zero_shot_results, keyword_scores):
    """Map all BERT results to the 9 motivational categories using multi-label hybrid approach."""
    # Calculate weighted scores for each category with adjusted weights and priors
    category_scores = {}

    # Adjusted ensemble weights to emphasize zero-shot and keywords, reduce dilution
    w_zero_shot = 0.5
    w_keywords = 0.3
    w_emotion = 0.15
    w_sentiment = 0.05

    for category in REFINED_MOTIVATION_KEYWORDS.keys():
        score = 0.0

        # 1. Zero-shot classification
        if zero_shot_results and 'all_scores' in zero_shot_results:
            z = zero_shot_results['all_scores'].get(category, 0.0)
            score += z * w_zero_shot

        # 2. Hybrid keyword matching (normalize by a soft cap)
        kw_raw = keyword_scores.get(category, 0.0)
        normalized_kw = 1.0 - np.exp(-kw_raw / 5.0)  # smooth saturation
        score += normalized_kw * w_keywords

        # 3. Emotion mapping
        if emotion_results:
            emotion_mapping = {
                'joy': ['Personal Development', 'Seeking Experiences'],
                'sadness': ['Close to the Heart', 'Altruism and Empathy'],
                'anger': ['Advocacy', 'Moral Obligation'],
                'fear': ['Close to the Heart', 'Stewardship'],
                'surprise': ['Seeking Experiences', 'Social Standing'],
                'disgust': ['Advocacy', 'Moral Obligation'],
                'trust': ['Stewardship', 'Altruism and Empathy'],
                'anticipation': ['Personal Development', 'Seeking Experiences']
            }
            e_sum = 0.0
            for emotion, emotion_score in emotion_results.items():
                if emotion in emotion_mapping and category in emotion_mapping[emotion]:
                    e_sum += emotion_score
            score += (e_sum * w_emotion)

        # 4. Sentiment mapping
        if sentiment_results and 'scores' in sentiment_results:
            sentiment_mapping = {
                'positive': ['Personal Development', 'Seeking Experiences', 'Social Standing'],
                'negative': ['Close to the Heart', 'Altruism and Empathy', 'Moral Obligation'],
                'neutral': ['Stewardship', 'Close to Home', 'Advocacy']
            }
            s_sum = 0.0
            for sentiment, sentiment_score in sentiment_results['scores'].items():
                if sentiment in sentiment_mapping and category in sentiment_mapping[sentiment]:
                    s_sum += sentiment_score
            score += (s_sum * w_sentiment)

        # Apply category prior to reduce bias
        prior = CATEGORY_PRIORS.get(category, 1.0)
        score *= prior

        category_scores[category] = score

    # Multi-label classification: Use sigmoid instead of softmax
    if category_scores:
        cats = list(category_scores.keys())
        vals = np.array([category_scores[c] for c in cats], dtype=float)
        
        # Apply sigmoid activation for multi-label (each category independent)
        # Scale by temperature for sharper separation
        scaled_vals = vals / max(TEMPERATURE, 1e-6)
        sigmoid_probs = 1.0 / (1.0 + np.exp(-scaled_vals))
        
        # Find categories above threshold
        above_threshold = [(cat, prob) for cat, prob in zip(cats, sigmoid_probs) 
                          if prob >= MULTI_LABEL_THRESHOLD]
        
        # Sort by probability and take top MAX_CATEGORIES_PER_STORY
        above_threshold.sort(key=lambda x: x[1], reverse=True)
        selected_categories = above_threshold[:MAX_CATEGORIES_PER_STORY]
        
        if selected_categories:
            # Primary category is the highest confidence one
            primary_category = selected_categories[0][0]
            primary_confidence = selected_categories[0][1]
            
            # All selected categories with their confidences
            all_scores = {c: {'score': float(category_scores[c]), 'prob': float(p)} 
                         for c, p in zip(cats, sigmoid_probs)}
            
            # Multi-label info
            multi_label_info = {
                'primary_category': primary_category,
                'primary_confidence': primary_confidence,
                'all_categories': [cat for cat, _ in selected_categories],
                'all_confidences': [conf for _, conf in selected_categories],
                'num_categories': len(selected_categories)
            }
        else:
            # No categories above threshold
            primary_category = 'No Category'
            primary_confidence = 0.0
            all_scores = {c: {'score': float(category_scores[c]), 'prob': float(p)} 
                         for c, p in zip(cats, sigmoid_probs)}
            multi_label_info = {
                'primary_category': 'No Category',
                'primary_confidence': 0.0,
                'all_categories': [],
                'all_confidences': [],
                'num_categories': 0
            }
    else:
        primary_category = 'No Category'
        primary_confidence = 0.0
        all_scores = {}
        multi_label_info = {
            'primary_category': 'No Category',
            'primary_confidence': 0.0,
            'all_categories': [],
            'all_confidences': [],
            'num_categories': 0
        }

    return {
        'dominant_category': primary_category,  # Keep for backward compatibility
        'confidence': primary_confidence,       # Keep for backward compatibility
        'all_scores': all_scores,
        'multi_label': multi_label_info
    }

def load_data_from_csv(csv_file, story_column='story', id_column='shortname', sample_size=None):
    """Load data from CSV file for analysis."""
    print(f"📂 Loading data from {csv_file}...")
    
    df = pd.read_csv(csv_file)
    
    # Check required columns
    if story_column not in df.columns:
        raise ValueError(f"Column '{story_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Rename columns to match expected format
    if id_column in df.columns:
        df = df.rename(columns={id_column: 'shortname'})
    elif 'shortname' not in df.columns and 'short_name' in df.columns:
        df = df.rename(columns={'short_name': 'shortname'})
    elif 'shortname' not in df.columns:
        df['shortname'] = df.index.astype(str)
    
    df = df.rename(columns={story_column: 'story'})
    
    # Filter out empty stories
    df = df[df['story'].notna() & (df['story'].str.len() > 50)].copy()
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} stories from CSV")
    return df


def run_hybrid_bert_analysis_from_csv(input_csv, story_column='story', id_column='shortname', sample_size=None):
    """Run hybrid BERT analysis from CSV file."""
    print("🚀 Starting Hybrid BERT Analysis")
    print("=" * 60)
    print(f"📊 Using Hybrid Approach with refined keywords")
    print(f"📁 Input file: {input_csv}")
    print("=" * 60)
    
    # Load models
    models = load_bert_models()
    
    # Load data from CSV
    df = load_data_from_csv(input_csv, story_column, id_column, sample_size)
    
    # Preprocess texts
    print("🔄 Preprocessing texts...")
    df['clean_story'] = df['story'].apply(preprocess_text)
    
    # Filter out very short texts
    df = df[df['clean_story'].str.len() >= 10].copy()

    # Deduplicate stories by normalized clean text (casefold + whitespace squeeze)
    print("🧹 Deduplicating stories...")
    norm = df['clean_story'].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    before = len(df)
    df = df.loc[~norm.duplicated()].copy()
    after = len(df)
    print(f"✅ Deduplicated: removed {before - after} duplicates; remaining {after} unique stories")
    print(f"✅ After preprocessing: {len(df)} stories")
    
    # Analyze stories
    print("🔄 Running hybrid BERT analysis...")
    results = []
    
    # Precompile keyword patterns for efficient matching
    keyword_patterns = _compile_keyword_patterns(REFINED_MOTIVATION_KEYWORDS)

    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(df)} stories...")
        
        text = row['clean_story']
        
        # Run all analyses
        emotion_results = analyze_emotions_bert(text, models['emotion'])
        sentiment_results = analyze_sentiment_bert(text, models['sentiment'])
        zero_shot_results = analyze_motivation_zero_shot(text, models['zero_shot'])
        keyword_scores = calculate_hybrid_keyword_scores(text, keyword_patterns)
        
        # Map to motivation category using hybrid approach
        motivation_results = map_to_motivation_category_hybrid(
            emotion_results, sentiment_results, zero_shot_results, keyword_scores
        )
        
        results.append({
            'short_name': row['shortname'],
            'clean_story': text,
            'story_length': len(text.split()),
            'emotions': emotion_results,
            'sentiment': sentiment_results,
            'zero_shot': zero_shot_results,
            'keyword_scores': keyword_scores,
            'dominant_category': motivation_results['dominant_category'],
            'category_confidence': motivation_results['confidence'],
            'all_category_scores': motivation_results['all_scores'],
            # Multi-label information
            'primary_category': motivation_results['multi_label']['primary_category'],
            'primary_confidence': motivation_results['multi_label']['primary_confidence'],
            'all_categories': motivation_results['multi_label']['all_categories'],
            'all_confidences': motivation_results['multi_label']['all_confidences'],
            'num_categories': motivation_results['multi_label']['num_categories']
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'bert_hybrid_results_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"✅ Results saved to: {output_file}")
    
    # Generate summary statistics
    print("\n📊 Multi-Label Hybrid BERT Analysis Summary")
    print("=" * 50)
    
    # Multi-label statistics
    print(f"\n🏷️ Multi-Label Statistics:")
    print(f"  Average categories per story: {results_df['num_categories'].mean():.2f}")
    print(f"  Stories with multiple categories: {(results_df['num_categories'] > 1).sum()}")
    print(f"  Stories with no categories: {(results_df['num_categories'] == 0).sum()}")
    
    # Primary category distribution
    print("\n🎯 Primary Category Distribution:")
    primary_counts = results_df['primary_category'].value_counts()
    for category, count in primary_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Confidence statistics (both primary and multi-label)
    print(f"\n📈 Confidence Statistics:")
    print(f"  Average primary confidence: {results_df['primary_confidence'].mean():.3f}")
    print(f"  Average legacy confidence: {results_df['category_confidence'].mean():.3f}")
    
    # High confidence examples
    high_conf_primary = results_df[results_df['primary_confidence'] > 0.7]
    high_conf_legacy = results_df[results_df['category_confidence'] > 0.7]
    print(f"\n🎯 High Confidence Examples (>0.7):")
    print(f"  Primary (multi-label): {len(high_conf_primary)} stories")
    print(f"  Legacy (softmax): {len(high_conf_legacy)} stories")
    
    # Show multi-label examples
    print(f"\n📋 Multi-Label Examples:")
    multi_label_examples = results_df[results_df['num_categories'] > 1].head(5)
    for _, example in multi_label_examples.iterrows():
        categories = example['all_categories']
        confidences = example['all_confidences']
        print(f"    • {example['short_name']}: {len(categories)} categories")
        for cat, conf in zip(categories, confidences):
            print(f"      - {cat}: {conf:.3f}")
        print(f"      Text: {example['clean_story'][:80]}...")
        print()
    
    return results_df

if __name__ == "__main__":
    print("🧠 Multi-Label Hybrid BERT Analysis for Crowdfunding Motivations")
    print("=" * 70)
    print("📊 Using Multi-Label Approach with sigmoid activation")
    print("📊 Threshold: 0.55 (optimized), Max categories per story: 5")
    print("=" * 70)
    print("
⚠️  This script requires a CSV file as input.")
    print("   Example usage:")
    print("   from bert_hybrid_analysis import run_hybrid_bert_analysis_from_csv")
    print("   results = run_hybrid_bert_analysis_from_csv('your_data.csv')")
    print("=" * 70)
    import sys
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
        results = run_hybrid_bert_analysis_from_csv(csv_file, sample_size=sample_size)
    else:
        print("
❌ Please provide a CSV file as argument: python bert_hybrid_analysis.py your_data.csv")
        sys.exit(1)
    
    print("\n✅ Hybrid BERT analysis completed successfully!")
    print(f"📁 Results saved in current directory")
    print(f"📊 Total stories analyzed: {len(results)}")
