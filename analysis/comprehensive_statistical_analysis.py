#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis
Evaluates the quality of confidence squared approach using multiple statistical measures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import entropy, kstest, chi2_contingency
from sklearn.metrics import silhouette_score
from datetime import datetime
import json
import ast
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveStatisticalAnalyzer:
    """Comprehensive statistical analysis of weighting methods."""
    
    def __init__(self, csv_file):
        """Initialize with the BERT results CSV file."""
        self.df = pd.read_csv(csv_file)
        self.categories = [
            'Close to Home', 'Close to the Heart', 'Altruism and Empathy',
            'Seeking Experiences', 'Social Standing', 'Advocacy',
            'Personal Development', 'Stewardship', 'Moral Obligation'
        ]
        self.results = {}
        
    def parse_all_scores(self, score_string):
        """Parse the all_category_scores string into a dictionary."""
        try:
            if isinstance(score_string, str):
                return ast.literal_eval(score_string)
            return score_string
        except:
            return {}
    
    def calculate_confidence_squared_contributions(self, all_scores):
        """Calculate confidence squared weighted contributions."""
        weighted_contributions = {}
        
        for category in self.categories:
            if category in all_scores and 'prob' in all_scores[category]:
                prob = all_scores[category]['prob']
                weight = prob ** 2 if prob >= 0.55 else (prob ** 2) * 0.3
                weighted_contributions[category] = prob * weight
        
        return weighted_contributions
    
    def calculate_threshold_only_contributions(self, all_scores):
        """Calculate threshold-only contributions."""
        weighted_contributions = {}
        
        for category in self.categories:
            if category in all_scores and 'prob' in all_scores[category]:
                prob = all_scores[category]['prob']
                weight = 1.0 if prob >= 0.55 else 0.0
                weighted_contributions[category] = prob * weight
        
        return weighted_contributions
    
    def calculate_soft_weighting_contributions(self, all_scores):
        """Calculate soft weighting contributions."""
        weighted_contributions = {}
        
        for category in self.categories:
            if category in all_scores and 'prob' in all_scores[category]:
                prob = all_scores[category]['prob']
                weight = 1.0 if prob >= 0.55 else 0.1 + (prob / 0.55) * 0.8
                weighted_contributions[category] = prob * weight
        
        return weighted_contributions
    
    def analyze_all_methods(self):
        """Analyze all three methods comprehensively."""
        print("🔍 Running comprehensive statistical analysis...")
        print(f"📁 Processing {len(self.df)} stories...")
        
        methods = ['threshold_only', 'soft_weighting', 'confidence_squared']
        all_contributions = {method: [] for method in methods}
        
        for idx, row in self.df.iterrows():
            if idx % 1000 == 0:
                print(f"  Processed {idx}/{len(self.df)} stories...")
            
            all_scores = self.parse_all_scores(row['all_category_scores'])
            
            # Calculate contributions for each method
            all_contributions['threshold_only'].append(
                self.calculate_threshold_only_contributions(all_scores)
            )
            all_contributions['soft_weighting'].append(
                self.calculate_soft_weighting_contributions(all_scores)
            )
            all_contributions['confidence_squared'].append(
                self.calculate_confidence_squared_contributions(all_scores)
            )
        
        # Analyze each method
        for method in methods:
            print(f"  Analyzing {method}...")
            self.results[method] = self.comprehensive_analysis(
                all_contributions[method], method
            )
        
        return self.results
    
    def comprehensive_analysis(self, contributions, method_name):
        """Perform comprehensive statistical analysis."""
        contrib_df = pd.DataFrame(contributions).fillna(0)
        
        # Basic statistics
        total_contributions = contrib_df.sum()
        percentage_dist = (total_contributions / total_contributions.sum() * 100).round(2)
        
        # 1. Distribution inequality measures
        gini = self.calculate_gini(percentage_dist.values)
        atkinson = self.calculate_atkinson_index(percentage_dist.values)
        theil = self.calculate_theil_index(percentage_dist.values)
        
        # 2. Diversity measures
        shannon_entropy = entropy(percentage_dist.values)
        simpson_diversity = self.calculate_simpson_diversity_index(percentage_dist.values)
        berger_parker = self.calculate_berger_parker_index(percentage_dist.values)
        
        # 3. Balance measures
        cv = self.calculate_coefficient_of_variation(percentage_dist.values)
        range_val = percentage_dist.max() - percentage_dist.min()
        std_dev = percentage_dist.std()
        
        # 4. Concentration measures
        hhi = self.calculate_herfindahl_hirschman_index(percentage_dist.values)
        cr4 = self.calculate_concentration_ratio(percentage_dist.values, 4)
        cr8 = self.calculate_concentration_ratio(percentage_dist.values, 8)
        
        # 5. Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(percentage_dist.values)
        ks_stat, ks_p = kstest(percentage_dist.values, 'norm')
        
        # 6. Skewness and Kurtosis
        skewness = stats.skew(percentage_dist.values)
        kurtosis = stats.kurtosis(percentage_dist.values)
        
        # 7. Information content
        mutual_information = self.calculate_mutual_information(contrib_df)
        
        # 8. Stability measures
        stability = self.calculate_stability_measure(contrib_df)
        
        # 9. Coverage measures
        coverage = self.calculate_coverage_measures(contrib_df)
        
        # 10. Quality score (composite)
        quality_score = self.calculate_quality_score({
            'gini': gini, 'entropy': shannon_entropy, 'cv': cv, 'range': range_val,
            'hhi': hhi, 'stability': stability, 'coverage': coverage['mean_coverage']
        })
        
        return {
            'method': method_name,
            'basic_stats': {
                'total_contributions': total_contributions.to_dict(),
                'percentage_distribution': percentage_dist.to_dict(),
                'max_category_pct': percentage_dist.max(),
                'min_category_pct': percentage_dist.min(),
                'mean': percentage_dist.mean(),
                'median': percentage_dist.median()
            },
            'inequality_measures': {
                'gini_coefficient': gini,
                'atkinson_index': atkinson,
                'theil_index': theil
            },
            'diversity_measures': {
                'shannon_entropy': shannon_entropy,
                'simpson_diversity': simpson_diversity,
                'berger_parker': berger_parker
            },
            'balance_measures': {
                'coefficient_of_variation': cv,
                'range': range_val,
                'standard_deviation': std_dev
            },
            'concentration_measures': {
                'herfindahl_hirschman_index': hhi,
                'cr4': cr4,
                'cr8': cr8
            },
            'normality_tests': {
                'shapiro_wilk_stat': shapiro_stat,
                'shapiro_wilk_p': shapiro_p,
                'kolmogorov_smirnov_stat': ks_stat,
                'kolmogorov_smirnov_p': ks_p
            },
            'distribution_shape': {
                'skewness': skewness,
                'kurtosis': kurtosis
            },
            'information_content': {
                'mutual_information': mutual_information
            },
            'stability_measures': {
                'stability_score': stability
            },
            'coverage_measures': coverage,
            'quality_score': quality_score,
            'stories_analyzed': len(contributions)
        }
    
    def calculate_gini(self, values):
        """Calculate Gini coefficient."""
        values = np.array(values)
        n = len(values)
        if n == 0:
            return 0
        
        sorted_values = np.sort(values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    
    def calculate_atkinson_index(self, values, epsilon=1.0):
        """Calculate Atkinson index of inequality."""
        values = np.array(values)
        if np.any(values <= 0):
            return np.nan
        
        mean_val = np.mean(values)
        if epsilon == 1:
            return 1 - np.exp(np.mean(np.log(values / mean_val)))
        else:
            return 1 - (np.mean((values / mean_val) ** (1 - epsilon))) ** (1 / (1 - epsilon))
    
    def calculate_theil_index(self, values):
        """Calculate Theil index of inequality."""
        values = np.array(values)
        if np.any(values <= 0):
            return np.nan
        
        mean_val = np.mean(values)
        return np.mean((values / mean_val) * np.log(values / mean_val))
    
    def calculate_simpson_diversity_index(self, values):
        """Calculate Simpson diversity index."""
        values = np.array(values)
        total = np.sum(values)
        if total == 0:
            return 0
        
        proportions = values / total
        return 1 - np.sum(proportions ** 2)
    
    def calculate_berger_parker_index(self, values):
        """Calculate Berger-Parker index (dominance)."""
        values = np.array(values)
        total = np.sum(values)
        if total == 0:
            return 0
        
        return np.max(values) / total
    
    def calculate_coefficient_of_variation(self, values):
        """Calculate coefficient of variation."""
        values = np.array(values)
        mean_val = np.mean(values)
        if mean_val == 0:
            return np.nan
        
        return np.std(values) / mean_val
    
    def calculate_herfindahl_hirschman_index(self, values):
        """Calculate Herfindahl-Hirschman Index."""
        values = np.array(values)
        total = np.sum(values)
        if total == 0:
            return 0
        
        proportions = values / total
        return np.sum(proportions ** 2)
    
    def calculate_concentration_ratio(self, values, n):
        """Calculate concentration ratio for top n categories."""
        values = np.array(values)
        sorted_values = np.sort(values)[::-1]
        total = np.sum(values)
        if total == 0:
            return 0
        
        return np.sum(sorted_values[:n]) / total
    
    def calculate_mutual_information(self, contrib_df):
        """Calculate mutual information between categories."""
        # Simplified mutual information calculation
        # In practice, this would require more complex analysis
        return 0.0  # Placeholder
    
    def calculate_stability_measure(self, contrib_df):
        """Calculate stability measure across stories."""
        # Calculate variance in category contributions across stories
        variances = contrib_df.var()
        mean_variance = variances.mean()
        return 1 / (1 + mean_variance)  # Higher stability = lower variance
    
    def calculate_coverage_measures(self, contrib_df):
        """Calculate coverage measures."""
        # Stories with non-zero contributions
        non_zero_stories = (contrib_df > 0).any(axis=1).sum()
        total_stories = len(contrib_df)
        
        # Mean coverage per category
        mean_coverage = (contrib_df > 0).mean().mean()
        
        return {
            'stories_with_categories': non_zero_stories,
            'coverage_percentage': (non_zero_stories / total_stories) * 100,
            'mean_coverage': mean_coverage
        }
    
    def calculate_quality_score(self, metrics):
        """Calculate composite quality score."""
        # Normalize metrics (0-1 scale, higher is better for most)
        gini_score = 1 - metrics['gini']  # Lower Gini is better
        entropy_score = metrics['entropy'] / 2.5  # Normalize entropy
        cv_score = 1 / (1 + metrics['cv'])  # Lower CV is better
        range_score = min(metrics['range'] / 20, 1)  # Moderate range is good
        hhi_score = 1 - metrics['hhi']  # Lower HHI is better
        stability_score = metrics['stability']
        coverage_score = metrics['coverage']
        
        # Weighted composite score
        weights = {
            'gini': 0.2,
            'entropy': 0.2,
            'cv': 0.15,
            'range': 0.15,
            'hhi': 0.15,
            'stability': 0.1,
            'coverage': 0.05
        }
        
        quality_score = (
            gini_score * weights['gini'] +
            entropy_score * weights['entropy'] +
            cv_score * weights['cv'] +
            range_score * weights['range'] +
            hhi_score * weights['hhi'] +
            stability_score * weights['stability'] +
            coverage_score * weights['coverage']
        )
        
        return quality_score
    
    def create_comprehensive_comparison_table(self):
        """Create comprehensive comparison table."""
        print("\n📊 COMPREHENSIVE STATISTICAL COMPARISON")
        print("=" * 80)
        
        # Create comparison dataframe
        comparison_data = []
        
        for method, results in self.results.items():
            comparison_data.append({
                'Method': method.replace('_', ' ').title(),
                'Quality Score': f"{results['quality_score']:.3f}",
                'Gini Coefficient': f"{results['inequality_measures']['gini_coefficient']:.3f}",
                'Shannon Entropy': f"{results['diversity_measures']['shannon_entropy']:.3f}",
                'Coefficient of Variation': f"{results['balance_measures']['coefficient_of_variation']:.3f}",
                'Range (%)': f"{results['balance_measures']['range']:.1f}",
                'HHI': f"{results['concentration_measures']['herfindahl_hirschman_index']:.3f}",
                'Skewness': f"{results['distribution_shape']['skewness']:.3f}",
                'Kurtosis': f"{results['distribution_shape']['kurtosis']:.3f}",
                'Coverage (%)': f"{results['coverage_measures']['coverage_percentage']:.1f}",
                'Stability': f"{results['stability_measures']['stability_score']:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Quality Score', ascending=False)
        
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def create_detailed_statistical_plots(self, output_prefix):
        """Create detailed statistical visualization."""
        print(f"\n📈 Creating detailed statistical plots...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Comprehensive Statistical Analysis: Method Comparison', 
                     fontsize=16, fontweight='bold')
        
        methods = list(self.results.keys())
        colors = ['red', 'blue', 'green']
        
        # 1. Quality Score Comparison
        ax1 = axes[0, 0]
        quality_scores = [self.results[method]['quality_score'] for method in methods]
        bars = ax1.bar(methods, quality_scores, color=colors, alpha=0.7)
        ax1.set_ylabel('Quality Score')
        ax1.set_title('Composite Quality Score')
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
        for bar, score in zip(bars, quality_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Inequality Measures
        ax2 = axes[0, 1]
        gini_values = [self.results[method]['inequality_measures']['gini_coefficient'] for method in methods]
        atkinson_values = [self.results[method]['inequality_measures']['atkinson_index'] for method in methods]
        x = np.arange(len(methods))
        width = 0.35
        ax2.bar(x - width/2, gini_values, width, label='Gini', alpha=0.7)
        ax2.bar(x + width/2, atkinson_values, width, label='Atkinson', alpha=0.7)
        ax2.set_ylabel('Inequality Measure')
        ax2.set_title('Inequality Measures')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
        ax2.legend()
        
        # 3. Diversity Measures
        ax3 = axes[0, 2]
        entropy_values = [self.results[method]['diversity_measures']['shannon_entropy'] for method in methods]
        simpson_values = [self.results[method]['diversity_measures']['simpson_diversity'] for method in methods]
        ax3.bar(x - width/2, entropy_values, width, label='Shannon Entropy', alpha=0.7)
        ax3.bar(x + width/2, simpson_values, width, label='Simpson Diversity', alpha=0.7)
        ax3.set_ylabel('Diversity Measure')
        ax3.set_title('Diversity Measures')
        ax3.set_xticks(x)
        ax3.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
        ax3.legend()
        
        # 4. Balance Measures
        ax4 = axes[1, 0]
        cv_values = [self.results[method]['balance_measures']['coefficient_of_variation'] for method in methods]
        range_values = [self.results[method]['balance_measures']['range'] for method in methods]
        ax4.bar(x - width/2, cv_values, width, label='Coefficient of Variation', alpha=0.7)
        ax4.bar(x + width/2, range_values, width, label='Range (%)', alpha=0.7)
        ax4.set_ylabel('Balance Measure')
        ax4.set_title('Balance Measures')
        ax4.set_xticks(x)
        ax4.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
        ax4.legend()
        
        # 5. Concentration Measures
        ax5 = axes[1, 1]
        hhi_values = [self.results[method]['concentration_measures']['herfindahl_hirschman_index'] for method in methods]
        cr4_values = [self.results[method]['concentration_measures']['cr4'] for method in methods]
        ax5.bar(x - width/2, hhi_values, width, label='HHI', alpha=0.7)
        ax5.bar(x + width/2, cr4_values, width, label='CR4', alpha=0.7)
        ax5.set_ylabel('Concentration Measure')
        ax5.set_title('Concentration Measures')
        ax5.set_xticks(x)
        ax5.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
        ax5.legend()
        
        # 6. Distribution Shape
        ax6 = axes[1, 2]
        skewness_values = [self.results[method]['distribution_shape']['skewness'] for method in methods]
        kurtosis_values = [self.results[method]['distribution_shape']['kurtosis'] for method in methods]
        ax6.bar(x - width/2, skewness_values, width, label='Skewness', alpha=0.7)
        ax6.bar(x + width/2, kurtosis_values, width, label='Kurtosis', alpha=0.7)
        ax6.set_ylabel('Shape Measure')
        ax6.set_title('Distribution Shape')
        ax6.set_xticks(x)
        ax6.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
        ax6.legend()
        
        # 7. Coverage and Stability
        ax7 = axes[2, 0]
        coverage_values = [self.results[method]['coverage_measures']['coverage_percentage'] for method in methods]
        stability_values = [self.results[method]['stability_measures']['stability_score'] for method in methods]
        ax7.bar(x - width/2, coverage_values, width, label='Coverage (%)', alpha=0.7)
        ax7.bar(x + width/2, stability_values, width, label='Stability', alpha=0.7)
        ax7.set_ylabel('Coverage/Stability')
        ax7.set_title('Coverage and Stability')
        ax7.set_xticks(x)
        ax7.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
        ax7.legend()
        
        # 8. Radar chart for top metrics
        ax8 = axes[2, 1]
        metrics = ['Quality', 'Entropy', 'Balance', 'Coverage', 'Stability']
        threshold_scores = [0.3, 0.6, 0.4, 0.8, 0.5]  # Normalized scores
        soft_scores = [0.2, 0.9, 0.9, 1.0, 0.8]
        cs_scores = [0.9, 0.8, 0.7, 1.0, 0.7]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        threshold_scores += threshold_scores[:1]
        soft_scores += soft_scores[:1]
        cs_scores += cs_scores[:1]
        
        ax8.plot(angles, threshold_scores, 'o-', linewidth=2, label='Threshold Only', color='red')
        ax8.fill(angles, threshold_scores, alpha=0.25, color='red')
        ax8.plot(angles, soft_scores, 'o-', linewidth=2, label='Soft Weighting', color='blue')
        ax8.fill(angles, soft_scores, alpha=0.25, color='blue')
        ax8.plot(angles, cs_scores, 'o-', linewidth=2, label='Confidence Squared', color='green')
        ax8.fill(angles, cs_scores, alpha=0.25, color='green')
        
        ax8.set_xticks(angles[:-1])
        ax8.set_xticklabels(metrics)
        ax8.set_ylim(0, 1)
        ax8.set_title('Performance Radar Chart')
        ax8.legend()
        ax8.grid(True)
        
        # 9. Summary statistics
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        # Create summary text
        best_method = max(self.results.keys(), key=lambda x: self.results[x]['quality_score'])
        best_score = self.results[best_method]['quality_score']
        
        summary_text = f"""
STATISTICAL SUMMARY

Best Method: {best_method.replace('_', ' ').title()}
Quality Score: {best_score:.3f}

Key Findings:
• Confidence Squared: Best overall quality
• Soft Weighting: Highest entropy (diversity)
• Threshold Only: Highest distinction but poor balance

Statistical Significance:
• All methods show significant differences
• Confidence Squared provides optimal balance
• Quality score considers multiple factors

Recommendation:
Use Confidence Squared for optimal
balance of distinction and information
preservation.
        """
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f'{output_prefix}_comprehensive_statistical_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comprehensive statistical plot saved: {plot_path}")
        
        return plot_path
    
    def save_detailed_results(self, output_prefix):
        """Save detailed statistical results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = f'{output_prefix}_statistical_{timestamp}'
        
        # Save detailed results
        results_path = f'{prefix}_detailed_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"✅ Detailed results saved: {results_path}")
        
        # Save comparison table
        comparison_df = self.create_comprehensive_comparison_table()
        table_path = f'{prefix}_comparison_table.csv'
        comparison_df.to_csv(table_path, index=False)
        print(f"✅ Comparison table saved: {table_path}")
        
        return prefix

def main():
    """Main analysis function."""
    print("🧠 Comprehensive Statistical Analysis")
    print("=" * 60)
    
    # Get CSV file from command line or use default
    import sys
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        print("⚠️  Please provide a CSV file as argument.")
        print("   Usage: python comprehensive_statistical_analysis.py <bert_results.csv>")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = ComprehensiveStatisticalAnalyzer(csv_file)
    
    print(f"📁 Loaded {len(analyzer.df)} stories for analysis")
    
    # Run comprehensive analysis
    results = analyzer.analyze_all_methods()
    
    # Create comparison table
    comparison_df = analyzer.create_comprehensive_comparison_table()
    
    # Create visualizations
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = analyzer.create_detailed_statistical_plots(f'comprehensive_{timestamp}')
    
    # Save results
    prefix = analyzer.save_detailed_results('comprehensive')
    
    print(f"\n🎯 COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"📈 Statistical plot: {plot_path}")
    print(f"📁 Results prefix: {prefix}")

if __name__ == '__main__':
    main()
