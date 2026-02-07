"""
Analysis script to compare preference-based and standard methods results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(pref_file='experiment_results_preference_based.csv',
                std_file='experiment_results_standard_methods.csv'):
    """Load both result files."""
    print("Loading results...")
    
    try:
        pref_df = pd.read_csv(pref_file)
        print(f"  Preference-based: {len(pref_df)} rows")
    except FileNotFoundError:
        print(f"  Warning: {pref_file} not found")
        pref_df = None
    
    try:
        std_df = pd.read_csv(std_file)
        print(f"  Standard methods: {len(std_df)} rows")
    except FileNotFoundError:
        print(f"  Warning: {std_file} not found")
        std_df = None
    
    return pref_df, std_df


def analyze_preference_based(pref_df):
    """Analyze preference-based method results."""
    if pref_df is None:
        print("\nNo preference-based results to analyze")
        return
    
    print("\n" + "=" * 80)
    print("PREFERENCE-BASED METHOD ANALYSIS")
    print("=" * 80)
    
    # Count unique experiments
    experiments = pref_df.groupby(['sample_idx', 'target_idx']).first().reset_index()
    n_experiments = len(experiments)
    
    print(f"\nTotal experiments: {n_experiments}")
    
    # Success rate
    successful = experiments[experiments['total_cf_found'] > 0]
    success_rate = len(successful) / n_experiments * 100
    print(f"Success rate: {success_rate:.1f}% ({len(successful)}/{n_experiments})")
    
    # Average counterfactuals found
    avg_cf = experiments['total_cf_found'].mean()
    print(f"Average CFs found per experiment: {avg_cf:.1f}")
    
    # Preference scores
    if 'cf_preference_score' in pref_df.columns:
        valid_scores = pref_df[pref_df['cf_preference_score'].notna()]['cf_preference_score']
        if len(valid_scores) > 0:
            print(f"\nPreference scores:")
            print(f"  Mean: {valid_scores.mean():.4f}")
            print(f"  Median: {valid_scores.median():.4f}")
            print(f"  Min: {valid_scores.min():.4f}")
            print(f"  Max: {valid_scores.max():.4f}")
    
    # Distance from target
    if 'cf_distance_from_target' in pref_df.columns:
        valid_dists = pref_df[pref_df['cf_distance_from_target'].notna()]['cf_distance_from_target']
        if len(valid_dists) > 0:
            print(f"\nDistance from target:")
            print(f"  Mean: {valid_dists.mean():.4f}")
            print(f"  Median: {valid_dists.median():.4f}")
            print(f"  Min: {valid_dists.min():.4f}")
            print(f"  Max: {valid_dists.max():.4f}")
    
    # Best results per experiment
    print(f"\nBest CF per experiment (top-ranked):")
    best_cfs = pref_df[pref_df['cf_rank'] == 1]
    if len(best_cfs) > 0:
        print(f"  Count: {len(best_cfs)}")
        print(f"  Avg preference score: {best_cfs['cf_preference_score'].mean():.4f}")
        print(f"  Avg distance from target: {best_cfs['cf_distance_from_target'].mean():.4f}")


def analyze_standard_methods(std_df):
    """Analyze standard methods results."""
    if std_df is None:
        print("\nNo standard methods results to analyze")
        return
    
    print("\n" + "=" * 80)
    print("STANDARD METHODS ANALYSIS")
    print("=" * 80)
    
    # Count unique experiments
    experiments = std_df.groupby(['sample_idx', 'target_idx', 'method']).first().reset_index()
    n_experiments = len(std_df.groupby(['sample_idx', 'target_idx']).first())
    methods = std_df['method'].unique()
    
    print(f"\nTotal experiments: {n_experiments}")
    print(f"Methods tested: {len(methods)} ({', '.join(methods)})")
    
    # Success rate by method
    print(f"\n{'Method':<20} {'Success Rate':<15} {'Valid CFs':<10} {'Total':<10}")
    print("-" * 60)
    
    method_stats = {}
    for method in methods:
        method_df = std_df[std_df['method'] == method]
        valid_count = method_df['valid'].sum()
        total_count = len(method_df)
        success_rate = valid_count / total_count * 100 if total_count > 0 else 0
        
        print(f"{method:<20} {success_rate:>6.1f}% {valid_count:>14} {total_count:>14}")
        method_stats[method] = {
            'valid_count': valid_count,
            'total_count': total_count,
            'success_rate': success_rate
        }
    
    # Distance metrics (only for valid CFs)
    valid_df = std_df[std_df['valid'] == True].copy()
    
    if len(valid_df) > 0:
        print(f"\n{'Method':<20} {'L2 Distance':<15} {'Sparsity':<15} {'Pred Error':<15}")
        print("-" * 70)
        
        for method in methods:
            method_valid = valid_df[valid_df['method'] == method]
            if len(method_valid) > 0:
                avg_l2 = method_valid['l2_distance'].mean()
                avg_sparsity = method_valid['sparsity'].mean()
                avg_error = method_valid['prediction_error'].mean()
                
                print(f"{method:<20} {avg_l2:>10.4f} {avg_sparsity:>15.2f} {avg_error:>15.4f}")
            else:
                print(f"{method:<20} {'N/A':>10} {'N/A':>15} {'N/A':>15}")
        
        # Overall statistics
        print(f"\n{'OVERALL':<20} {valid_df['l2_distance'].mean():>10.4f} "
              f"{valid_df['sparsity'].mean():>15.2f} {valid_df['prediction_error'].mean():>15.4f}")


def compare_methods(pref_df, std_df):
    """Compare preference-based vs standard methods."""
    if pref_df is None or std_df is None:
        print("\nCannot compare - missing data")
        return
    
    print("\n" + "=" * 80)
    print("COMPARISON: PREFERENCE-BASED vs STANDARD METHODS")
    print("=" * 80)
    
    # Get experiment pairs
    pref_experiments = pref_df.groupby(['sample_idx', 'target_idx']).first().reset_index()
    std_experiments = std_df.groupby(['sample_idx', 'target_idx']).first().reset_index()
    
    # Success rates
    pref_success = (pref_experiments['total_cf_found'] > 0).sum()
    pref_total = len(pref_experiments)
    pref_rate = pref_success / pref_total * 100 if pref_total > 0 else 0
    
    std_success = (std_df['valid'] == True).sum()
    std_total = len(std_df)
    std_rate = std_success / std_total * 100 if std_total > 0 else 0
    
    print(f"\nSuccess Rates:")
    print(f"  Preference-based: {pref_rate:.1f}% ({pref_success}/{pref_total} experiments)")
    print(f"  Standard methods: {std_rate:.1f}% ({std_success}/{std_total} attempts)")
    
    # Count experiments where each method succeeded
    print(f"\nPer-experiment comparison:")
    
    # Get unique sample-target pairs
    pairs = pref_experiments[['sample_idx', 'target_idx']].drop_duplicates()
    
    both_success = 0
    only_pref = 0
    only_std = 0
    both_fail = 0
    
    for _, pair in pairs.iterrows():
        sample_idx = pair['sample_idx']
        target_idx = pair['target_idx']
        
        # Check preference success
        pref_pair = pref_experiments[(pref_experiments['sample_idx'] == sample_idx) & 
                                     (pref_experiments['target_idx'] == target_idx)]
        pref_found = len(pref_pair) > 0 and pref_pair['total_cf_found'].values[0] > 0
        
        # Check standard methods success (at least one method succeeded)
        std_pair = std_df[(std_df['sample_idx'] == sample_idx) & 
                         (std_df['target_idx'] == target_idx)]
        std_found = len(std_pair) > 0 and (std_pair['valid'] == True).any()
        
        if pref_found and std_found:
            both_success += 1
        elif pref_found and not std_found:
            only_pref += 1
        elif not pref_found and std_found:
            only_std += 1
        else:
            both_fail += 1
    
    total_pairs = len(pairs)
    print(f"  Both succeeded: {both_success}/{total_pairs} ({both_success/total_pairs*100:.1f}%)")
    print(f"  Only preference: {only_pref}/{total_pairs} ({only_pref/total_pairs*100:.1f}%)")
    print(f"  Only standard: {only_std}/{total_pairs} ({only_std/total_pairs*100:.1f}%)")
    print(f"  Both failed: {both_fail}/{total_pairs} ({both_fail/total_pairs*100:.1f}%)")


def create_visualizations(pref_df, std_df, output_dir='.'):
    """Create comparison visualizations."""
    if pref_df is None and std_df is None:
        print("\nNo data for visualizations")
        return
    
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Success rate comparison
    if std_df is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        method_success = []
        method_names = []
        
        for method in std_df['method'].unique():
            method_df = std_df[std_df['method'] == method]
            success_rate = (method_df['valid'] == True).sum() / len(method_df) * 100
            method_success.append(success_rate)
            method_names.append(method.replace('_', ' ').title())
        
        # Add preference-based if available
        if pref_df is not None:
            pref_experiments = pref_df.groupby(['sample_idx', 'target_idx']).first()
            pref_rate = (pref_experiments['total_cf_found'] > 0).sum() / len(pref_experiments) * 100
            method_success.append(pref_rate)
            method_names.append('Preference-Based')
        
        bars = ax.bar(method_names, method_success, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(method_names)])
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Counterfactual Generation Success Rate by Method', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'success_rate_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: success_rate_comparison.png")
        plt.close()
    
    # 2. Distance comparison (only valid CFs)
    if std_df is not None:
        valid_std = std_df[std_df['valid'] == True].copy()
        
        if len(valid_std) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            methods = []
            distances = []
            
            for method in valid_std['method'].unique():
                method_data = valid_std[valid_std['method'] == method]['l2_distance']
                methods.extend([method.replace('_', ' ').title()] * len(method_data))
                distances.extend(method_data)
            
            df_plot = pd.DataFrame({'Method': methods, 'L2 Distance': distances})
            sns.boxplot(data=df_plot, x='Method', y='L2 Distance', ax=ax)
            ax.set_title('L2 Distance Distribution by Method (Valid CFs Only)', fontsize=14, fontweight='bold')
            ax.set_ylabel('L2 Distance', fontsize=12)
            ax.set_xlabel('')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / 'distance_comparison.png', dpi=300, bbox_inches='tight')
            print(f"  Saved: distance_comparison.png")
            plt.close()
    
    # 3. Sparsity comparison
    if std_df is not None:
        valid_std = std_df[std_df['valid'] == True].copy()
        
        if len(valid_std) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            method_sparsity = valid_std.groupby('method')['sparsity'].mean().sort_values()
            bars = ax.barh(range(len(method_sparsity)), method_sparsity.values)
            ax.set_yticks(range(len(method_sparsity)))
            ax.set_yticklabels([m.replace('_', ' ').title() for m in method_sparsity.index])
            ax.set_xlabel('Average Number of Changed Features', fontsize=12)
            ax.set_title('Feature Sparsity by Method (Valid CFs Only)', fontsize=14, fontweight='bold')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.2f}',
                       ha='left', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'sparsity_comparison.png', dpi=300, bbox_inches='tight')
            print(f"  Saved: sparsity_comparison.png")
            plt.close()
    
    print(f"\nVisualizations saved to: {output_dir.absolute()}")


def main():
    """Main analysis function."""
    print("=" * 80)
    print("COUNTERFACTUAL METHODS COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Load data
    pref_df, std_df = load_results()
    
    # Analyze each method
    analyze_preference_based(pref_df)
    analyze_standard_methods(std_df)
    
    # Compare methods
    compare_methods(pref_df, std_df)
    
    # Create visualizations
    try:
        create_visualizations(pref_df, std_df)
    except Exception as e:
        print(f"\nWarning: Could not create visualizations: {e}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
