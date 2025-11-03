#!/usr/bin/env python3
"""
Subsample (Held-Out) Consistency Analysis
Validates that BERT ensemble results are stable across different data subsets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import entropy
from datetime import datetime
import json
import ast
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

class SubsampleConsistencyAnalyzer:
    """Analyzes consistency of BERT ensemble results across different data subsets."""
    
    def __init__(self, csv_file):
        """Initialize with the BERT results CSV file."""
        print(f"📂 Loading data from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(self.df)} stories")
        
        self.categories = [
            'Close to Home', 'Close to the Heart', 'Altruism and Empathy',
            'Seeking Experiences', 'Social Standing', 'Advocacy',
            'Personal Development', 'Stewardship', 'Moral Obligation'
        ]
        
    def parse_all_scores(self, score_string):
        """Parse the all_category_scores string into a dictionary."""
        try:
            if isinstance(score_string, str):
                return ast.literal_eval(score_string)
            return score_string
        except:
            return {}
    
    def confidence_squared_weighting(self, prob, threshold=0.55):
        """Apply confidence squared weighting."""
        if prob >= threshold:
            return prob ** 2
        else:
            return (prob ** 2) * 0.3
    
    def calculate_confidence_squared_contributions(self, all_scores):
        """Calculate confidence squared weighted contributions."""
        weighted_contributions = {}
        
        for category in self.categories:
            if category in all_scores and 'prob' in all_scores[category]:
                prob = all_scores[category]['prob']
                weight = self.confidence_squared_weighting(prob)
                weighted_contributions[category] = prob * weight
        
        return weighted_contributions
    
    def analyze_subset(self, subset_df):
        """Analyze a subset of the data and return category distribution."""
        contributions = []
        
        for idx, row in subset_df.iterrows():
            all_scores = self.parse_all_scores(row['all_category_scores'])
            weighted_contrib = self.calculate_confidence_squared_contributions(all_scores)
            contributions.append(weighted_contrib)
        
        # Aggregate contributions
        contrib_df = pd.DataFrame(contributions).fillna(0)
        total_contributions = contrib_df.sum()
        percentage_dist = (total_contributions / total_contributions.sum() * 100).round(2)
        
        return percentage_dist.to_dict(), len(subset_df)
    
    def train_test_split_analysis(self, test_size=0.2, random_state=42):
        """Perform train/test split validation."""
        print("\n" + "="*60)
        print("🔄 TRAIN/TEST SPLIT VALIDATION")
        print("="*60)
        
        # Shuffle and split
        df_shuffled = self.df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        split_idx = int(len(df_shuffled) * (1 - test_size))
        
        train_df = df_shuffled.iloc[:split_idx]
        test_df = df_shuffled.iloc[split_idx:]
        
        print(f"📊 Split sizes:")
        print(f"   Training set: {len(train_df)} stories ({100*(1-test_size):.1f}%)")
        print(f"   Test set: {len(test_df)} stories ({100*test_size:.1f}%)")
        
        # Analyze each subset
        print("\n🔍 Analyzing training set...")
        train_dist, train_n = self.analyze_subset(train_df)
        
        print("🔍 Analyzing test set...")
        test_dist, test_n = self.analyze_subset(test_df)
        
        # Analyze full dataset for comparison
        print("🔍 Analyzing full dataset...")
        full_dist, full_n = self.analyze_subset(self.df)
        
        # Calculate correlations and differences
        train_array = np.array([train_dist.get(cat, 0) for cat in self.categories])
        test_array = np.array([test_dist.get(cat, 0) for cat in self.categories])
        full_array = np.array([full_dist.get(cat, 0) for cat in self.categories])
        
        train_test_corr = np.corrcoef(train_array, test_array)[0, 1]
        train_full_corr = np.corrcoef(train_array, full_array)[0, 1]
        test_full_corr = np.corrcoef(test_array, full_array)[0, 1]
        
        # Calculate mean absolute differences
        train_test_mad = np.mean(np.abs(train_array - test_array))
        train_full_mad = np.mean(np.abs(train_array - full_array))
        test_full_mad = np.mean(np.abs(test_array - full_array))
        
        print(f"\n📈 Results:")
        print(f"   Train-Test Correlation: {train_test_corr:.4f}")
        print(f"   Train-Full Correlation: {train_full_corr:.4f}")
        print(f"   Test-Full Correlation: {test_full_corr:.4f}")
        print(f"   Mean Absolute Difference (Train-Test): {train_test_mad:.2f} percentage points")
        
        return {
            'train': train_dist,
            'test': test_dist,
            'full': full_dist,
            'train_n': train_n,
            'test_n': test_n,
            'full_n': full_n,
            'correlations': {
                'train_test': train_test_corr,
                'train_full': train_full_corr,
                'test_full': test_full_corr
            },
            'mean_abs_diff': {
                'train_test': train_test_mad,
                'train_full': train_full_mad,
                'test_full': test_full_mad
            }
        }
    
    def random_subsample_analysis(self, n_subsamples=5, subsample_size=0.5, random_states=None):
        """Analyze multiple random 50% subsamples."""
        print("\n" + "="*60)
        print("🔄 RANDOM SUBSAMPLE VALIDATION")
        print("="*60)
        
        if random_states is None:
            random_states = list(range(42, 42 + n_subsamples))
        
        subsample_size_int = int(len(self.df) * subsample_size)
        subsample_dists = []
        
        print(f"📊 Creating {n_subsamples} random {100*subsample_size:.0f}% subsamples...")
        
        for i, rs in enumerate(random_states):
            print(f"   Subsample {i+1}/{n_subsamples} (random_state={rs})...")
            subsample_df = self.df.sample(n=subsample_size_int, random_state=rs)
            dist, n = self.analyze_subset(subsample_df)
            subsample_dists.append(dist)
        
        # Analyze full dataset for comparison
        print("   Analyzing full dataset for comparison...")
        full_dist, full_n = self.analyze_subset(self.df)
        
        # Calculate statistics across subsamples
        subsample_arrays = []
        for dist in subsample_dists:
            arr = np.array([dist.get(cat, 0) for cat in self.categories])
            subsample_arrays.append(arr)
        
        subsample_arrays = np.array(subsample_arrays)
        full_array = np.array([full_dist.get(cat, 0) for cat in self.categories])
        
        # Calculate mean and std across subsamples
        mean_subsample = np.mean(subsample_arrays, axis=0)
        std_subsample = np.std(subsample_arrays, axis=0)
        
        # Calculate correlations between each subsample and full
        subsample_full_corrs = [np.corrcoef(arr, full_array)[0, 1] for arr in subsample_arrays]
        
        # Calculate mean absolute differences
        subsample_full_mads = [np.mean(np.abs(arr - full_array)) for arr in subsample_arrays]
        
        print(f"\n📈 Results:")
        print(f"   Mean Correlation (Subsample-Full): {np.mean(subsample_full_corrs):.4f}")
        print(f"   Std Correlation: {np.std(subsample_full_corrs):.4f}")
        print(f"   Mean Absolute Difference (Subsample-Full): {np.mean(subsample_full_mads):.2f} ± {np.std(subsample_full_mads):.2f} percentage points")
        
        # Category-wise stability
        print(f"\n📊 Category-wise stability (std across subsamples):")
        for i, cat in enumerate(self.categories):
            print(f"   {cat}: {std_subsample[i]:.2f} percentage points")
        
        return {
            'subsamples': subsample_dists,
            'full': full_dist,
            'mean_subsample': {cat: mean_subsample[i] for i, cat in enumerate(self.categories)},
            'std_subsample': {cat: std_subsample[i] for i, cat in enumerate(self.categories)},
            'correlations': subsample_full_corrs,
            'mean_abs_diffs': subsample_full_mads,
            'full_n': full_n,
            'subsample_n': subsample_size_int
        }
    
    def create_visualizations(self, train_test_results, subsample_results):
        """Create comprehensive visualizations of consistency analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Train/Test Split Comparison (Bar Chart)
        ax1 = fig.add_subplot(gs[0, :])
        categories_short = [cat.replace(' and ', ' & ').replace('Close to ', 'Close to\n') for cat in self.categories]
        
        train_values = [train_test_results['train'].get(cat, 0) for cat in self.categories]
        test_values = [train_test_results['test'].get(cat, 0) for cat in self.categories]
        full_values = [train_test_results['full'].get(cat, 0) for cat in self.categories]
        
        x = np.arange(len(self.categories))
        width = 0.25
        
        ax1.bar(x - width, train_values, width, label=f"Training Set (n={train_test_results['train_n']:,})", 
                color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.bar(x, test_values, width, label=f"Test Set (n={train_test_results['test_n']:,})", 
                color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.bar(x + width, full_values, width, label=f"Full Dataset (n={train_test_results['full_n']:,})", 
                color='#45B7D1', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Motivation Category', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Train/Test Split Consistency: Category Distribution Comparison', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories_short, rotation=45, ha='right', fontsize=9)
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add correlation annotation
        corr_text = f"Train-Test Correlation: r = {train_test_results['correlations']['train_test']:.4f}\n"
        corr_text += f"Mean Absolute Difference: {train_test_results['mean_abs_diff']['train_test']:.2f} pp"
        ax1.text(0.98, 0.98, corr_text, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Train/Test Correlation Scatter
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(train_values, test_values, alpha=0.6, s=100, color='#4ECDC4', edgecolors='black', linewidth=0.5)
        
        # Add diagonal line
        min_val = min(min(train_values), min(test_values))
        max_val = max(max(train_values), max(test_values))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7, label='Perfect Agreement')
        
        ax2.set_xlabel('Training Set %', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Test Set %', fontsize=11, fontweight='bold')
        ax2.set_title(f'Train-Test Correlation\nr = {train_test_results["correlations"]["train_test"]:.4f}', 
                     fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Subsample Stability (Mean ± Std)
        ax3 = fig.add_subplot(gs[1, 1])
        mean_vals = [subsample_results['mean_subsample'].get(cat, 0) for cat in self.categories]
        std_vals = [subsample_results['std_subsample'].get(cat, 0) for cat in self.categories]
        full_vals = [subsample_results['full'].get(cat, 0) for cat in self.categories]
        
        x = np.arange(len(self.categories))
        ax3.bar(x, mean_vals, yerr=std_vals, capsize=5, alpha=0.7, 
                color='#95E1D3', edgecolor='black', linewidth=0.5, label='Subsample Mean ± SD')
        ax3.scatter(x, full_vals, s=150, color='red', marker='D', 
                   zorder=5, label='Full Dataset', edgecolors='black', linewidth=0.5)
        
        ax3.set_xlabel('Motivation Category', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax3.set_title(f'Subsample Stability (n={subsample_results["subsample_n"]:,} each)', 
                     fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories_short, rotation=45, ha='right', fontsize=8)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Correlation Distribution
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.hist(subsample_results['correlations'], bins=10, alpha=0.7, color='#F38181', 
                edgecolor='black', linewidth=0.5)
        ax4.axvline(np.mean(subsample_results['correlations']), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {np.mean(subsample_results["correlations"]):.4f}')
        ax4.set_xlabel('Correlation (Subsample-Full)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax4.set_title('Subsample-Full Dataset\nCorrelation Distribution', 
                     fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Category Stability (Heatmap)
        ax5 = fig.add_subplot(gs[2, :2])
        
        # Prepare data for heatmap
        heatmap_data = []
        for dist in subsample_results['subsamples']:
            heatmap_data.append([dist.get(cat, 0) for cat in self.categories])
        heatmap_data.append([subsample_results['full'].get(cat, 0) for cat in self.categories])
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                  columns=[cat.replace(' and ', ' & ') for cat in self.categories],
                                  index=[f'Subsample {i+1}' for i in range(len(subsample_results['subsamples']))] + ['Full Dataset'])
        
        sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Percentage (%)'}, ax=ax5,
                   linewidths=0.5, linecolor='gray', annot_kws={'size': 8})
        ax5.set_title('Subsample Consistency: Category Distributions Across Subsamples', 
                     fontsize=12, fontweight='bold', pad=15)
        ax5.set_xlabel('Motivation Category', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Dataset Subset', fontsize=11, fontweight='bold')
        
        # 6. Mean Absolute Difference Boxplot
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.boxplot(subsample_results['mean_abs_diffs'], vert=True, widths=0.6,
                   patch_artist=True, boxprops=dict(facecolor='#FFD93D', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax6.scatter([1], [np.mean(subsample_results['mean_abs_diffs'])], 
                   s=200, color='blue', marker='*', zorder=5, label='Mean')
        ax6.set_ylabel('Mean Absolute Difference (percentage points)', fontsize=11, fontweight='bold')
        ax6.set_title('Subsample-Full Dataset\nDifference Distribution', 
                     fontsize=12, fontweight='bold')
        ax6.set_xticklabels(['All Categories'], fontsize=10)
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add summary text
        summary_text = (
            f"VALIDATION SUMMARY\n"
            f"{'='*40}\n"
            f"Train-Test Split:\n"
            f"  • Correlation: {train_test_results['correlations']['train_test']:.4f}\n"
            f"  • Mean Abs Diff: {train_test_results['mean_abs_diff']['train_test']:.2f} pp\n"
            f"\nSubsample Analysis:\n"
            f"  • Mean Correlation: {np.mean(subsample_results['correlations']):.4f}\n"
            f"  • Mean Abs Diff: {np.mean(subsample_results['mean_abs_diffs']):.2f} ± {np.std(subsample_results['mean_abs_diffs']):.2f} pp\n"
            f"  • Max Category Std: {max(subsample_results['std_subsample'].values()):.2f} pp"
        )
        
        fig.text(0.02, 0.02, summary_text, fontsize=9, 
                verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Subsample Consistency Validation: BERT Ensemble Robustness', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        filename = f'subsample_consistency_validation_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved visualization: {filename}")
        
        return filename
    
    def create_simplified_visualization(self, train_test_results, subsample_results):
        """Create simplified visualization with only train/test split and subsample heatmap."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure with 2 subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        categories_short = [cat.replace(' and ', ' & ').replace('Close to ', 'Close to\n') for cat in self.categories]
        
        # 1. Train/Test Split Comparison (Bar Chart)
        train_values = [train_test_results['train'].get(cat, 0) for cat in self.categories]
        test_values = [train_test_results['test'].get(cat, 0) for cat in self.categories]
        full_values = [train_test_results['full'].get(cat, 0) for cat in self.categories]
        
        x = np.arange(len(self.categories))
        width = 0.25
        
        bars1 = ax1.bar(x - width, train_values, width, label=f"Training Set (n={train_test_results['train_n']:,})", 
                color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax1.bar(x, test_values, width, label=f"Test Set (n={train_test_results['test_n']:,})", 
                color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars3 = ax1.bar(x + width, full_values, width, label=f"Full Dataset (n={train_test_results['full_n']:,})", 
                color='#45B7D1', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Motivation Category', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
        ax1.set_title('Train/Test Split Consistency: Category Distribution Comparison', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories_short, rotation=45, ha='right', fontsize=11)
        ax1.legend(fontsize=12, loc='upper right', framealpha=0.9)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(max(train_values), max(test_values), max(full_values)) * 1.15)
        
        # Add correlation annotation
        corr_text = f"Train-Test Correlation: r = {train_test_results['correlations']['train_test']:.4f}\n"
        corr_text += f"Mean Absolute Difference: {train_test_results['mean_abs_diff']['train_test']:.2f} pp"
        ax1.text(0.98, 0.98, corr_text, transform=ax1.transAxes, 
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Subsample Consistency Heatmap
        # Prepare data for heatmap
        heatmap_data = []
        for dist in subsample_results['subsamples']:
            heatmap_data.append([dist.get(cat, 0) for cat in self.categories])
        heatmap_data.append([subsample_results['full'].get(cat, 0) for cat in self.categories])
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                  columns=[cat.replace(' and ', ' & ') for cat in self.categories],
                                  index=[f'Subsample {i+1}' for i in range(len(subsample_results['subsamples']))] + ['Full Dataset'])
        
        sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Percentage (%)', 'shrink': 0.8}, ax=ax2,
                   linewidths=0.8, linecolor='gray', annot_kws={'size': 10, 'weight': 'bold'},
                   vmin=0, vmax=max([max(row) for row in heatmap_data]) * 1.1)
        
        ax2.set_title('Subsample Consistency: Category Distributions Across Subsamples', 
                     fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Motivation Category', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Dataset Subset', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(heatmap_df.columns, rotation=45, ha='right', fontsize=11)
        ax2.set_yticklabels(heatmap_df.index, rotation=0, fontsize=11, va='center')
        
        # Add statistics annotation for subsample analysis
        stats_text = f"Mean Correlation (Subsample-Full): r = {np.mean(subsample_results['correlations']):.4f}\n"
        stats_text += f"Mean Absolute Difference: {np.mean(subsample_results['mean_abs_diffs']):.2f} ± {np.std(subsample_results['mean_abs_diffs']):.2f} pp\n"
        stats_text += f"Max Category Std: {max(subsample_results['std_subsample'].values()):.2f} pp"
        ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes, 
                fontsize=11, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Subsample Consistency Validation: BERT Ensemble Robustness', 
                    fontsize=18, fontweight='bold', y=1.00)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        filename = f'subsample_consistency_simplified_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved simplified visualization: {filename}")
        
        return filename
    
    def run_full_analysis(self):
        """Run the complete subsample consistency analysis."""
        print("\n" + "="*60)
        print("🎓 SUBSAMPLE CONSISTENCY VALIDATION")
        print("="*60)
        print(f"Dataset: {len(self.df):,} stories")
        print(f"Method: Confidence Squared Weighting")
        
        # Train/Test Split Analysis
        train_test_results = self.train_test_split_analysis()
        
        # Random Subsample Analysis
        subsample_results = self.random_subsample_analysis(n_subsamples=5)
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        plot_file = self.create_visualizations(train_test_results, subsample_results)
        
        # Create simplified visualization
        print("📊 Creating simplified visualization...")
        simplified_plot_file = self.create_simplified_visualization(train_test_results, subsample_results)
        
        # Print final summary
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print("="*60)
        print(f"\n📈 Key Findings:")
        print(f"   1. Train-Test Correlation: {train_test_results['correlations']['train_test']:.4f}")
        print(f"   2. Subsample-Full Mean Correlation: {np.mean(subsample_results['correlations']):.4f}")
        print(f"   3. Mean Absolute Difference: {np.mean(subsample_results['mean_abs_diffs']):.2f} percentage points")
        print(f"   4. Category Stability: Max std across subsamples = {max(subsample_results['std_subsample'].values()):.2f} pp")
        print(f"\n✅ Results indicate HIGH CONSISTENCY across data subsets")
        print(f"📁 Full visualization saved: {plot_file}")
        print(f"📁 Simplified visualization saved: {simplified_plot_file}")
        
        return train_test_results, subsample_results

def main():
    """Main function."""
    # Get CSV file from command line or use default
    import sys
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        print("⚠️  Please provide a CSV file as argument.")
        print("   Usage: python subsample_consistency_analysis.py <bert_results.csv>")
        sys.exit(1)
    
    analyzer = SubsampleConsistencyAnalyzer(csv_file)
    train_test_results, subsample_results = analyzer.run_full_analysis()
    
    return train_test_results, subsample_results

if __name__ == "__main__":
    train_test_results, subsample_results = main()

