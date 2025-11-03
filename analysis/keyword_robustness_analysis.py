#!/usr/bin/env python3
"""
Keyword Component Robustness Analysis
Validates that keyword dictionaries are not dominated by single keywords

ACADEMIC RELEASE VERSION: This script reads from CSV files instead of databases.
No credential information is required.

Input Requirements:
- BERT results CSV with 'clean_story' column (output from bert_hybrid_analysis.py)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Keyword dictionaries (matching bert_hybrid_analysis.py)
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

# Specificity weights
SPECIFICITY_WEIGHTS = {
    # High specificity (1.0)
    'marathon training': 1.0, 'charity run': 1.0, 'fundraising event': 1.0, 'sponsored walk': 1.0,
    'donation management': 1.0, 'fund allocation': 1.0, 'charity oversight': 1.0, 'financial stewardship': 1.0,
    'local community': 1.0, 'our neighborhood': 1.0, 'nearby school': 1.0, 'regional hospital': 1.0,
    'raise awareness for': 1.0, 'campaign to support': 1.0, 'advocate for change': 1.0,
    'in memory of': 1.0, 'close to my heart': 1.0, 'personal connection': 1.0,
    'challenge myself': 1.0, 'personal challenge': 1.0, 'charity event': 1.0,
    'personal growth': 1.0, 'learn new skills': 1.0, 'develop myself': 1.0,
    'like and share': 1.0, 'friends support': 1.0, 'social media': 1.0,
    
    # Medium specificity (0.7)
    'community project': 0.7, 'local charity': 0.7, 'neighborhood initiative': 0.7,
    'transparent giving': 0.7, 'responsible management': 0.7, 'efficient allocation': 0.7,
    'policy reform': 0.7, 'equality rights': 0.7, 'social justice': 0.7,
    'family member': 0.7, 'loved one': 0.7, 'personal experience': 0.7,
    'fundraising challenge': 0.7, 'sponsored activity': 0.7, 'exciting challenge': 0.7,
    'skill development': 0.7, 'learning journey': 0.7, 'self-improvement': 0.7,
    'social recognition': 0.7, 'identity expression': 0.7, 'social influence': 0.7,
    'caring for others': 0.7, 'support those in need': 0.7, 'change lives': 0.7,
    'justice': 0.7, 'compassion': 0.7, 'humanity': 0.7,
    
    # Low specificity (0.3) - will be filtered by MIN_KEYWORD_WEIGHT
    'help others': 0.3, 'support': 0.3, 'help': 0.3, 'community': 0.3, 'local': 0.3,
    'challenge': 0.3, 'event': 0.3, 'experience': 0.3, 'growth': 0.3, 'development': 0.3,
    'family': 0.3, 'friend': 0.3, 'personal': 0.3, 'social': 0.3,
    
    # Generic (0.1) - will be filtered
    'fun': 0.1, 'giving': 0.1, 'use': 0.1, 'like': 0.1, 'share': 0.1, 'post': 0.1, 'follow': 0.1
}

MIN_KEYWORD_WEIGHT = 0.5  # Same as in bert_hybrid_analysis.py

def _compile_keyword_patterns(motivation_keywords):
    """Pre-compile regex patterns for keywords."""
    patterns = {}
    for category, keywords in motivation_keywords.items():
        compiled = []
        for kw in keywords:
            escaped = re.escape(kw)
            escaped = escaped.replace(r"\ ", r"\s+")
            pattern = rf"(?i)(?<!\w){escaped}(?!\w)"
            compiled.append(re.compile(pattern))
        patterns[category] = compiled
    return patterns

def get_keyword_weight(keyword):
    """Get specificity weight for a keyword."""
    # Try exact match first
    if keyword in SPECIFICITY_WEIGHTS:
        return SPECIFICITY_WEIGHTS[keyword]
    
    # Try case-insensitive match
    for k, w in SPECIFICITY_WEIGHTS.items():
        if keyword.lower() == k.lower():
            return w
    
    # Default weight
    return 0.5

def track_keyword_matches(text, keyword_patterns):
    """
    Track which specific keywords match in the text.
    Returns a dict: {category: {keyword: count}}
    """
    if not text:
        return {}
    
    category_matches = {}
    
    for category, patterns in keyword_patterns.items():
        keyword_counts = Counter()
        
        for pat in patterns:
            # Extract original keyword from pattern (approximate)
            kw_pattern = pat.pattern
            # Try to find the original keyword
            original_kw = None
            for kw in REFINED_MOTIVATION_KEYWORDS[category]:
                if re.escape(kw).replace(r"\ ", r"\s+") in kw_pattern:
                    original_kw = kw
                    break
            
            if original_kw is None:
                continue
            
            # Check weight threshold
            weight = get_keyword_weight(original_kw)
            if weight < MIN_KEYWORD_WEIGHT:
                continue
            
            # Count matches (capped at 3, same as in bert_hybrid_analysis.py)
            matches = len(pat.findall(text))
            if matches > 0:
                # Store the actual match count (before cap) for analysis
                keyword_counts[original_kw] += min(matches, 3)
        
        if keyword_counts:
            category_matches[category] = keyword_counts
    
    return category_matches

def calculate_gini_coefficient(values):
    """
    Calculate Gini coefficient for inequality measurement.
    Returns value between 0 (perfect equality) and 1 (perfect inequality).
    """
    if len(values) == 0 or sum(values) == 0:
        return 0.0
    
    values = np.array(sorted(values))
    n = len(values)
    cumsum = np.cumsum(values)
    
    # Gini = 2 * sum of (i * value_i) / (n * sum(values)) - (n + 1) / n
    gini = (2 * np.sum((np.arange(1, n + 1)) * values)) / (n * np.sum(values)) - (n + 1) / n
    
    return gini

def analyze_keyword_robustness(df, keyword_patterns):
    """Analyze keyword robustness for each category."""
    print("=" * 70)
    print("🔍 KEYWORD COMPONENT ROBUSTNESS ANALYSIS")
    print("=" * 70)
    
    # Aggregate keyword matches across all stories
    category_keyword_totals = defaultdict(Counter)
    category_total_matches = defaultdict(int)
    
    print(f"\n📊 Processing {len(df)} stories...")
    
    for idx, row in df.iterrows():
        if (idx + 1) % 1000 == 0:
            print(f"   Processed {idx + 1}/{len(df)} stories...")
        
        text = str(row.get('clean_story', ''))
        if not text or len(text.strip()) < 10:
            continue
        
        matches = track_keyword_matches(text, keyword_patterns)
        
        # Aggregate matches
        for category, keyword_counts in matches.items():
            for keyword, count in keyword_counts.items():
                category_keyword_totals[category][keyword] += count
                category_total_matches[category] += count
    
    print(f"✅ Processed all stories")
    
    # Calculate statistics per category
    results = []
    
    print(f"\n📊 Calculating robustness statistics...")
    for category in REFINED_MOTIVATION_KEYWORDS.keys():
        if category not in category_keyword_totals or category_total_matches[category] == 0:
            results.append({
                'category': category,
                'total_matches': 0,
                'num_keywords_used': 0,
                'most_frequent_keyword': None,
                'most_frequent_share': 0.0,
                'gini_coefficient': 0.0,
                'top_5_keywords': []
            })
            continue
        
        keyword_counts = category_keyword_totals[category]
        total = category_total_matches[category]
        
        # Most frequent keyword and its share
        if keyword_counts:
            most_frequent = keyword_counts.most_common(1)[0]
            most_frequent_share = (most_frequent[1] / total) * 100
            
            # Gini coefficient
            match_counts = list(keyword_counts.values())
            gini = calculate_gini_coefficient(match_counts)
            
            # Top 5 keywords
            top_5 = keyword_counts.most_common(5)
            
            results.append({
                'category': category,
                'total_matches': total,
                'num_keywords_used': len(keyword_counts),
                'most_frequent_keyword': most_frequent[0],
                'most_frequent_count': most_frequent[1],
                'most_frequent_share': most_frequent_share,
                'gini_coefficient': gini,
                'top_5_keywords': top_5
            })
        else:
            results.append({
                'category': category,
                'total_matches': 0,
                'num_keywords_used': 0,
                'most_frequent_keyword': None,
                'most_frequent_share': 0.0,
                'gini_coefficient': 0.0,
                'top_5_keywords': []
            })
    
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 KEYWORD ROBUSTNESS SUMMARY")
    print("=" * 70)
    
    for _, row in results_df.iterrows():
        if row['total_matches'] > 0:
            print(f"\n🏷️  {row['category']}:")
            print(f"   Total matches: {row['total_matches']:,}")
            print(f"   Keywords used: {row['num_keywords_used']}")
            print(f"   Most frequent: '{row['most_frequent_keyword']}' ({row['most_frequent_share']:.1f}%)")
            print(f"   Gini coefficient: {row['gini_coefficient']:.3f}")
    
    # Overall statistics
    valid_results = results_df[results_df['total_matches'] > 0]
    if len(valid_results) > 0:
        median_share = valid_results['most_frequent_share'].median()
        mean_gini = valid_results['gini_coefficient'].mean()
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Median most-frequent keyword share: {median_share:.1f}%")
        print(f"   Mean Gini coefficient: {mean_gini:.3f}")
        print(f"   Categories with Gini < 0.5: {len(valid_results[valid_results['gini_coefficient'] < 0.5])}/{len(valid_results)}")
    
    return results_df, category_keyword_totals

def create_visualizations(results_df, category_keyword_totals):
    """Create visualizations of keyword robustness."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Filter to categories with matches
    valid_results = results_df[results_df['total_matches'] > 0].copy()
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # 1. Most Frequent Keyword Share by Category
    ax1 = fig.add_subplot(gs[0, 0])
    categories = valid_results['category'].values
    shares = valid_results['most_frequent_share'].values
    
    bars = ax1.barh(range(len(categories)), shares, color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(categories)))
    ax1.set_yticklabels(categories, fontsize=10)
    ax1.set_xlabel('Share of Matches (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Most Frequent Keyword Share by Category', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # Add median line
    median_share = valid_results['most_frequent_share'].median()
    ax1.axvline(median_share, color='red', linestyle='--', linewidth=2, label=f'Median: {median_share:.1f}%')
    ax1.legend(fontsize=10)
    
    # Add value labels
    for i, (cat, share) in enumerate(zip(categories, shares)):
        ax1.text(share + 1, i, f'{share:.1f}%', va='center', fontsize=9)
    
    # 2. Gini Coefficient by Category
    ax2 = fig.add_subplot(gs[0, 1])
    ginis = valid_results['gini_coefficient'].values
    
    bars2 = ax2.barh(range(len(categories)), ginis, color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(categories)))
    ax2.set_yticklabels(categories, fontsize=10)
    ax2.set_xlabel('Gini Coefficient', fontsize=12, fontweight='bold')
    ax2.set_title('Keyword Usage Inequality (Gini Coefficient)', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.0)
    
    # Add reference lines
    ax2.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='Gini = 0.5 (moderate inequality)')
    mean_gini = valid_results['gini_coefficient'].mean()
    ax2.axvline(mean_gini, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_gini:.3f}')
    ax2.legend(fontsize=10)
    
    # Add value labels
    for i, (cat, gini) in enumerate(zip(categories, ginis)):
        ax2.text(gini + 0.02, i, f'{gini:.3f}', va='center', fontsize=9)
    
    # 3. Scatter: Most Frequent Share vs Gini
    ax3 = fig.add_subplot(gs[1, :])
    ax3.scatter(valid_results['most_frequent_share'], valid_results['gini_coefficient'], 
               s=200, alpha=0.7, color='#45B7D1', edgecolors='black', linewidth=1.5)
    
    # Add category labels
    for _, row in valid_results.iterrows():
        ax3.annotate(row['category'], 
                    (row['most_frequent_share'], row['gini_coefficient']),
                    fontsize=9, alpha=0.8)
    
    ax3.set_xlabel('Most Frequent Keyword Share (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Gini Coefficient', fontsize=12, fontweight='bold')
    ax3.set_title('Keyword Robustness: Share vs Inequality', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3)
    
    # Add reference zones
    ax3.axvline(30, color='gray', linestyle=':', alpha=0.5, label='30% share')
    ax3.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Gini = 0.5')
    ax3.legend(fontsize=10)
    
    # 4. Top Keywords Distribution (Heatmap for top categories)
    ax4 = fig.add_subplot(gs[2, :])
    
    # Select top 6 categories by total matches
    top_categories = valid_results.nlargest(6, 'total_matches')['category'].tolist()
    
    # Prepare heatmap data
    heatmap_data = []
    heatmap_categories = []
    max_keywords = 0
    
    for cat in top_categories:
        if cat in category_keyword_totals:
            top_keywords = category_keyword_totals[cat].most_common(10)
            if top_keywords:
                max_keywords = max(max_keywords, len(top_keywords))
                heatmap_categories.append(cat)
                heatmap_data.append([count for _, count in top_keywords])
    
    # Pad to same length
    for i in range(len(heatmap_data)):
        while len(heatmap_data[i]) < max_keywords:
            heatmap_data[i].append(0)
    
    # Get keyword labels for the most diverse category
    if heatmap_categories:
        most_diverse_cat = valid_results.loc[valid_results['category'].isin(heatmap_categories), 'gini_coefficient'].idxmin()
        most_diverse_cat_name = valid_results.loc[most_diverse_cat, 'category']
        top_keywords_labels = [kw for kw, _ in category_keyword_totals[most_diverse_cat_name].most_common(max_keywords)]
        
        # Create heatmap
        heatmap_df = pd.DataFrame(heatmap_data, index=heatmap_categories, columns=top_keywords_labels[:max_keywords])
        
        sns.heatmap(heatmap_df, annot=True, fmt='d', cmap='YlOrRd', ax=ax4,
                   cbar_kws={'label': 'Match Count'}, linewidths=0.5, annot_kws={'size': 8})
        ax4.set_title('Top Keyword Match Counts by Category\n(Top 6 Categories by Total Matches)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax4.set_xlabel('Keywords (ranked by frequency in most diverse category)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Category', fontsize=11, fontweight='bold')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.suptitle('Keyword Component Robustness Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    filename = f'keyword_robustness_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {filename}")
    
    return filename

def main(csv_file):
    """Main analysis function."""
    print(f"📂 Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df)} stories")
    
    # Check required column
    if 'clean_story' not in df.columns:
        raise ValueError("CSV file must contain 'clean_story' column. Please use output from bert_hybrid_analysis.py")
    
    # Compile keyword patterns
    print("\n🔄 Compiling keyword patterns...")
    keyword_patterns = _compile_keyword_patterns(REFINED_MOTIVATION_KEYWORDS)
    print("✅ Patterns compiled")
    
    # Analyze keyword robustness
    results_df, category_keyword_totals = analyze_keyword_robustness(df, keyword_patterns)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    create_visualizations(results_df, category_keyword_totals)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'keyword_robustness_results_{timestamp}.csv'
    
    # Prepare summary for CSV (flatten top_5_keywords)
    summary_data = []
    for _, row in results_df.iterrows():
        summary_row = {
            'category': row['category'],
            'total_matches': row['total_matches'],
            'num_keywords_used': row['num_keywords_used'],
            'most_frequent_keyword': row['most_frequent_keyword'],
            'most_frequent_count': row.get('most_frequent_count', 0),
            'most_frequent_share_pct': row['most_frequent_share'],
            'gini_coefficient': row['gini_coefficient']
        }
        
        # Add top 5 keywords as separate columns
        for i, (kw, count) in enumerate(row['top_5_keywords'][:5], 1):
            summary_row[f'top_{i}_keyword'] = kw
            summary_row[f'top_{i}_count'] = count
        
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    print(f"✅ Results saved: {output_file}")
    
    print("\n✅ Keyword Robustness Analysis Complete!")
    return results_df, category_keyword_totals

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️  Usage: python keyword_robustness_analysis.py <bert_results.csv>")
        print("   The CSV file must contain 'clean_story' column from bert_hybrid_analysis.py")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    results_df, category_keyword_totals = main(csv_file)

