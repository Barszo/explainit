"""
Comprehensive Comparison Analysis: Preference-based vs Standard Methods

This script:
1. Loads results from both preference-based and standard methods experiments
2. Calculates preference scores for standard methods' counterfactuals
3. Compares methods on multiple metrics (preference score, L2 distance, timing, etc.)
4. Generates comparison tables and visualizations
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import ast
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import experiment class to use preference functions
sys.path.append(str(Path(__file__).parent))
from experiment_final import AutoMPGExperiment


def load_preference_results(filepath='experiment_results_preference_based.csv'):
    """Load preference-based method results."""
    logger.info(f"Loading preference-based results from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Parse list columns - handle numpy arrays and regular lists
    def safe_parse_list(x):
        if pd.isna(x) or x == '':
            return None
        try:
            # Try direct eval first
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            # If that fails, try removing numpy array formatting
            import re
            # Remove numpy array wrapper if present
            x_clean = re.sub(r'array\((.*?)\)', r'\1', str(x))
            # Remove dtype specifications
            x_clean = re.sub(r',\s*dtype=\w+', '', x_clean)
            try:
                return ast.literal_eval(x_clean)
            except:
                # Last resort: try to extract numbers manually
                try:
                    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', str(x))
                    return [float(n) for n in numbers]
                except:
                    logger.warning(f"Could not parse value: {x}")
                    return None
    
    for col in ['sample_values', 'exemplar_values', 'cf_values']:
        if col in df.columns:
            df[col] = df[col].apply(safe_parse_list)
    
    logger.info(f"Loaded {len(df)} rows from preference-based results")
    return df


def load_standard_results(filepath='experiment_results_standard_methods.csv'):
    """Load standard methods results."""
    logger.info(f"Loading standard methods results from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Parse list columns - handle numpy arrays and regular lists
    def safe_parse_list(x):
        if pd.isna(x) or x == '':
            return None
        try:
            # Try direct eval first
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            # If that fails, try removing numpy array formatting
            import re
            # Remove numpy array wrapper if present
            x_clean = re.sub(r'array\((.*?)\)', r'\1', str(x))
            # Remove dtype specifications
            x_clean = re.sub(r',\s*dtype=\w+', '', x_clean)
            try:
                return ast.literal_eval(x_clean)
            except:
                # Last resort: try to extract numbers manually
                try:
                    numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', str(x))
                    return [float(n) for n in numbers]
                except:
                    logger.warning(f"Could not parse value: {x}")
                    return None
    
    for col in ['sample_values', 'cf_values']:
        if col in df.columns:
            df[col] = df[col].apply(safe_parse_list)
    
    logger.info(f"Loaded {len(df)} rows from standard methods results")
    return df


def calculate_preference_score_for_cf(sample, exemplar, cf, X_train, exemplar_weight=0.5, 
                                     penalty_mode='strict_rejection'):
    """
    Calculate preference score for a counterfactual using the same logic as in experiments.
    
    Args:
        sample: Original sample values (list)
        exemplar: Exemplar values (list)
        cf: Counterfactual values (list)
        X_train: Training data to extract min/max
        exemplar_weight: Weight for exemplar in preference function
        penalty_mode: How to handle constraint violations:
            - 'no_penalty': No constraint checking (original behavior)
            - 'strict_rejection': Return 0.0 if ANY constraint violated
            - 'per_feature_zero': Assign 0 for violating features, sum all
            - 'linear_penalty': Subtract distance from boundary
            - 'squared_penalty': Subtract squared distance from boundary
    
    Returns:
        Total preference score
    """
    from explainit.priorities.nonlinear import exponential
    
    experiment = AutoMPGExperiment()
    
    # Define preferences
    preferences = experiment.define_preferences(sample, exemplar, X_train, exemplar_weight)
    
    # Check for violations first (for all modes except no_penalty)
    violations = {}
    if penalty_mode != 'no_penalty':
        for idx, pref_info in preferences['numerical'].items():
            cf_value = cf[idx]
            acceptable_min = pref_info['min']
            acceptable_max = pref_info['max']
            
            if cf_value < acceptable_min:
                violations[idx] = ('below', acceptable_min - cf_value)
            elif cf_value > acceptable_max:
                violations[idx] = ('above', cf_value - acceptable_max)
    
    # Apply penalty based on mode
    if penalty_mode == 'strict_rejection' and violations:
        # If ANY constraint violated, return 0
        return 0.0
    
    # Calculate preference score
    total_score = 0.0
    for idx, pref_info in preferences['numerical'].items():
        cf_value = cf[idx]
        pref_func = pref_info['function']
        
        if penalty_mode == 'per_feature_zero' and idx in violations:
            # This feature violated constraints, assign 0
            feature_score = 0.0
        elif penalty_mode == 'linear_penalty' and idx in violations:
            # Subtract the violation distance
            feature_score = pref_func(cf_value) - violations[idx][1]
        elif penalty_mode == 'squared_penalty' and idx in violations:
            # Subtract the squared violation distance
            feature_score = pref_func(cf_value) - (violations[idx][1] ** 2)
        else:
            # No violation or no_penalty mode
            feature_score = pref_func(cf_value)
        
        total_score += feature_score
    
    return total_score


def add_preference_scores_to_standard_methods(df_standard, df_pref, X_train, exemplar_weight=0.5):
    """
    Calculate and add multiple types of preference scores for all valid standard method counterfactuals.
    
    Args:
        df_standard: DataFrame with standard methods results
        df_pref: DataFrame with preference-based results (to get exemplars)
        X_train: Training data for min/max extraction
        exemplar_weight: Weight for exemplar
    
    Returns:
        DataFrame with added preference_score columns
    """
    logger.info("Calculating preference scores for standard methods' counterfactuals...")
    
    # Create a mapping of (sample_idx, target_idx) -> exemplar from preference results
    exemplar_map = {}
    for _, row in df_pref.iterrows():
        key = (row['sample_idx'], row['target_idx'])
        if key not in exemplar_map and row['exemplar_values'] is not None:
            exemplar_map[key] = row['exemplar_values']
    
    logger.info(f"Found {len(exemplar_map)} sample-target pairs with exemplars")
    
    # Initialize score lists for different penalty modes
    score_modes = {
        'score_no_penalty': [],
        'score_strict_rejection': [],
        'score_per_feature_zero': [],
        'score_linear_penalty': [],
        'score_squared_penalty': []
    }
    
    for idx, row in df_standard.iterrows():
        if row['valid'] and row['cf_values'] is not None:
            # Get sample and exemplar from the mapping
            sample = row['sample_values']
            key = (row['sample_idx'], row['target_idx'])
            exemplar = exemplar_map.get(key)
            
            if exemplar is None:
                logger.warning(f"No exemplar found for sample {row['sample_idx']} → target {row['target_idx']}")
                for mode_key in score_modes:
                    score_modes[mode_key].append(np.nan)
                continue
            
            cf = row['cf_values']
            
            # Calculate all types of preference scores
            try:
                for mode_key, mode_name in [
                    ('score_no_penalty', 'no_penalty'),
                    ('score_strict_rejection', 'strict_rejection'),
                    ('score_per_feature_zero', 'per_feature_zero'),
                    ('score_linear_penalty', 'linear_penalty'),
                    ('score_squared_penalty', 'squared_penalty')
                ]:
                    score = calculate_preference_score_for_cf(
                        sample, exemplar, cf, X_train, exemplar_weight, 
                        penalty_mode=mode_name
                    )
                    score_modes[mode_key].append(score)
            except Exception as e:
                logger.warning(f"Failed to calculate preference scores for row {idx}: {e}")
                for mode_key in score_modes:
                    score_modes[mode_key].append(np.nan)
        else:
            for mode_key in score_modes:
                score_modes[mode_key].append(np.nan)
    
    # Add all score columns to dataframe
    for mode_key, scores in score_modes.items():
        df_standard[mode_key] = scores
    
    valid_count = len([s for s in score_modes['score_no_penalty'] if not np.isnan(s)])
    logger.info(f"Added {len(score_modes)} types of preference scores to {valid_count} valid counterfactuals")
    
    return df_standard


def get_best_cf_per_pair_preference(df_pref):
    """
    Get the best counterfactual (highest preference score) for each sample-target pair
    from preference-based method.
    """
    logger.info("Extracting best counterfactuals from preference-based method...")
    
    # Group by sample-target pair and get the first row (highest ranked)
    best_cfs = df_pref[df_pref['cf_rank'] == 1].copy()
    
    logger.info(f"Found {len(best_cfs)} best counterfactuals from preference method")
    return best_cfs


def get_valid_cfs_standard(df_std):
    """
    Get all valid counterfactuals from standard methods.
    """
    logger.info("Extracting valid counterfactuals from standard methods...")
    
    valid_cfs = df_std[df_std['valid'] == True].copy()
    
    logger.info(f"Found {len(valid_cfs)} valid counterfactuals from standard methods")
    return valid_cfs


def create_comparison_table(df_pref_best, df_std_valid):
    """
    Create a comprehensive comparison table.
    """
    logger.info("\n" + "=" * 100)
    logger.info("COMPARISON TABLE: PREFERENCE-BASED vs STANDARD METHODS")
    logger.info("=" * 100)
    
    # Get unique sample-target pairs
    pairs_pref = df_pref_best[['sample_idx', 'target_idx']].drop_duplicates()
    
    results = []
    
    for _, pair in pairs_pref.iterrows():
        sample_idx = pair['sample_idx']
        target_idx = pair['target_idx']
        
        # Get preference method result
        pref_row = df_pref_best[
            (df_pref_best['sample_idx'] == sample_idx) & 
            (df_pref_best['target_idx'] == target_idx)
        ]
        
        if len(pref_row) == 0:
            continue
        
        pref_row = pref_row.iloc[0]
        
        # Get standard methods results
        std_rows = df_std_valid[
            (df_std_valid['sample_idx'] == sample_idx) & 
            (df_std_valid['target_idx'] == target_idx)
        ]
        
        # Build comparison row
        row_data = {
            'Sample→Target': f"{sample_idx}→{target_idx}",
            'Sample Pred': f"{pref_row['sample_prediction']:.2f}",
            'Target Pred': f"{pref_row['target_prediction']:.2f}",
            'Distance': f"{abs(pref_row['sample_prediction'] - pref_row['target_prediction']):.2f}",
        }
        
        # Preference method
        row_data['Pref_Score'] = f"{pref_row['cf_preference_score']:.2f}"
        row_data['Pref_L2'] = f"{pref_row['cf_l2_distance']:.4f}"
        row_data['Pref_Sparsity'] = int(pref_row['cf_sparsity'])
        row_data['Pref_Time'] = f"{pref_row['computation_time']:.3f}s"
        
        # Standard methods
        for method in ['wachter', 'growing_spheres', 'prototype', 'gradient_based']:
            method_row = std_rows[std_rows['method'] == method]
            
            if len(method_row) > 0:
                method_row = method_row.iloc[0]
                # Use preference_score if available, otherwise mark as N/A
                pref_score = method_row.get('preference_score', np.nan)
                row_data[f'{method[:6]}_Score'] = f"{pref_score:.2f}" if not np.isnan(pref_score) else "N/A"
                row_data[f'{method[:6]}_L2'] = f"{method_row['l2_distance']:.4f}"
                row_data[f'{method[:6]}_Sparsity'] = int(method_row['sparsity'])
            else:
                row_data[f'{method[:6]}_Score'] = "FAILED"
                row_data[f'{method[:6]}_L2'] = "-"
                row_data[f'{method[:6]}_Sparsity'] = "-"
        
        # Add standard methods time (same for all methods in a pair)
        if len(std_rows) > 0:
            row_data['Std_Time'] = f"{std_rows.iloc[0]['computation_time']:.3f}s"
        else:
            row_data['Std_Time'] = "-"
        
        results.append(row_data)
    
    # Create DataFrame
    df_comparison = pd.DataFrame(results)
    
    # Print table
    print("\n" + df_comparison.to_string(index=False))
    
    return df_comparison


def create_score_comparison_table(df_pref_best, df_std_valid):
    """
    Create a focused table comparing preference scores across all methods and penalty modes.
    """
    logger.info("\n" + "=" * 120)
    logger.info("PREFERENCE SCORE COMPARISON TABLE")
    logger.info("=" * 120)
    
    # Get unique sample-target pairs
    pairs_pref = df_pref_best[['sample_idx', 'target_idx']].drop_duplicates()
    
    results = []
    
    for _, pair_row in pairs_pref.iterrows():
        sample_idx = pair_row['sample_idx']
        target_idx = pair_row['target_idx']
        
        # Get preference method result
        pref_row = df_pref_best[
            (df_pref_best['sample_idx'] == sample_idx) & 
            (df_pref_best['target_idx'] == target_idx)
        ]
        
        if len(pref_row) == 0:
            continue
        
        pref_row = pref_row.iloc[0]
        
        # Get standard methods results
        std_rows = df_std_valid[
            (df_std_valid['sample_idx'] == sample_idx) & 
            (df_std_valid['target_idx'] == target_idx)
        ]
        
        # Build comparison row
        row_data = {
            'Sample→Target': f"{sample_idx}→{target_idx}",
            'Pref_Method': f"{pref_row['cf_preference_score']:.2f}",
        }
        
        # Standard methods - all score types
        for method in ['wachter', 'growing_spheres', 'prototype', 'gradient_based']:
            method_short = method[:6]
            method_row = std_rows[std_rows['method'] == method]
            
            if len(method_row) > 0:
                method_row = method_row.iloc[0]
                
                # Add all score types
                row_data[f'{method_short}_NoPen'] = f"{method_row.get('score_no_penalty', np.nan):.2f}" if not np.isnan(method_row.get('score_no_penalty', np.nan)) else "N/A"
                row_data[f'{method_short}_Strict'] = f"{method_row.get('score_strict_rejection', np.nan):.2f}" if not np.isnan(method_row.get('score_strict_rejection', np.nan)) else "N/A"
                row_data[f'{method_short}_PerFeat'] = f"{method_row.get('score_per_feature_zero', np.nan):.2f}" if not np.isnan(method_row.get('score_per_feature_zero', np.nan)) else "N/A"
                row_data[f'{method_short}_Linear'] = f"{method_row.get('score_linear_penalty', np.nan):.2f}" if not np.isnan(method_row.get('score_linear_penalty', np.nan)) else "N/A"
                row_data[f'{method_short}_Squared'] = f"{method_row.get('score_squared_penalty', np.nan):.2f}" if not np.isnan(method_row.get('score_squared_penalty', np.nan)) else "N/A"
            else:
                row_data[f'{method_short}_NoPen'] = "FAILED"
                row_data[f'{method_short}_Strict'] = "FAILED"
                row_data[f'{method_short}_PerFeat'] = "FAILED"
                row_data[f'{method_short}_Linear'] = "FAILED"
                row_data[f'{method_short}_Squared'] = "FAILED"
        
        results.append(row_data)
    
    # Create DataFrame
    df_scores = pd.DataFrame(results)
    
    # Print table
    print("\n" + df_scores.to_string(index=False))
    
    # Save to CSV
    csv_filename = 'score_comparison_table.csv'
    df_scores.to_csv(csv_filename, index=False)
    logger.info(f"\nScore comparison table saved to {csv_filename}")
    
    # Print legend
    print("\n" + "=" * 120)
    print("SCORE TYPES LEGEND:")
    print("  Pref_Method: Preference-based method score (always satisfies constraints)")
    print("  NoPen:       No penalty - original score without constraint checking")
    print("  Strict:      Strict rejection - return 0 if ANY constraint violated")
    print("  PerFeat:     Per-feature zero - assign 0 for violating features, sum all")
    print("  Linear:      Linear penalty - subtract distance from constraint boundary")
    print("  Squared:     Squared penalty - subtract squared distance from boundary")
    print("=" * 120)
    
    return df_scores


def create_summary_statistics(df_pref_best, df_std_valid):
    """
    Create summary statistics comparing all methods.
    """
    logger.info("\n" + "=" * 100)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 100)
    
    # Success rates
    total_pairs = len(df_pref_best[['sample_idx', 'target_idx']].drop_duplicates())
    
    print(f"\n{'Method':<20} {'Success Rate':<15} {'Avg L2 Dist':<15} {'Avg Sparsity':<15} {'Avg Time':<15}")
    print("-" * 80)
    
    # Preference method
    pref_success = len(df_pref_best[df_pref_best['cf_rank'] == 1])
    pref_l2_mean = df_pref_best['cf_l2_distance'].mean()
    pref_sparsity_mean = df_pref_best['cf_sparsity'].mean()
    pref_time_mean = df_pref_best.groupby(['sample_idx', 'target_idx'])['computation_time'].first().mean()
    
    print(f"{'Preference-based':<20} {pref_success}/{total_pairs} ({100*pref_success/total_pairs:.1f}%)"
          f"{'':>2} {pref_l2_mean:.4f}{'':>7} {pref_sparsity_mean:.2f}{'':>10} {pref_time_mean:.3f}s")
    
    # Standard methods
    for method in ['wachter', 'growing_spheres', 'prototype', 'gradient_based']:
        method_rows = df_std_valid[df_std_valid['method'] == method]
        
        # Count unique sample-target pairs
        method_success = len(method_rows[['sample_idx', 'target_idx']].drop_duplicates())
        
        if len(method_rows) > 0:
            method_l2_mean = method_rows['l2_distance'].mean()
            method_sparsity_mean = method_rows['sparsity'].mean()
            method_time_mean = method_rows.groupby(['sample_idx', 'target_idx'])['computation_time'].first().mean()
            
            print(f"{method:<20} {method_success}/{total_pairs} ({100*method_success/total_pairs:.1f}%)"
                  f"{'':>2} {method_l2_mean:.4f}{'':>7} {method_sparsity_mean:.2f}{'':>10} {method_time_mean:.3f}s")
        else:
            print(f"{method:<20} {method_success}/{total_pairs} (0.0%)    -           -           -")
    
    print()
    
    # Preference score comparison (only for valid standard method CFs)
    if 'preference_score' in df_std_valid.columns:
        print(f"\n{'Method':<20} {'Avg Pref Score':<20} {'Max Pref Score':<20} {'Min Pref Score':<20}")
        print("-" * 80)
        
        pref_score_mean = df_pref_best['cf_preference_score'].mean()
        pref_score_max = df_pref_best['cf_preference_score'].max()
        pref_score_min = df_pref_best['cf_preference_score'].min()
        
        print(f"{'Preference-based':<20} {pref_score_mean:.4f}{'':>12} {pref_score_max:.4f}{'':>12} {pref_score_min:.4f}")
        
        for method in ['wachter', 'growing_spheres', 'prototype', 'gradient_based']:
            method_rows = df_std_valid[df_std_valid['method'] == method]
            
            if len(method_rows) > 0 and method_rows['preference_score'].notna().any():
                method_score_mean = method_rows['preference_score'].mean()
                method_score_max = method_rows['preference_score'].max()
                method_score_min = method_rows['preference_score'].min()
                
                print(f"{method:<20} {method_score_mean:.4f}{'':>12} {method_score_max:.4f}{'':>12} {method_score_min:.4f}")
            else:
                print(f"{method:<20} {'N/A':<20} {'N/A':<20} {'N/A':<20}")
        
        print()


def create_visualizations(df_pref_best, df_std_valid, output_dir='comparison_plots'):
    """
    Create comparison visualizations.
    """
    logger.info(f"Creating visualizations in {output_dir}/...")
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Success Rate Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    total_pairs = len(df_pref_best[['sample_idx', 'target_idx']].drop_duplicates())
    
    methods = ['Preference-based', 'Wachter', 'Growing Spheres', 'Prototype', 'Gradient-based']
    success_counts = []
    
    # Preference method
    pref_success = len(df_pref_best[df_pref_best['cf_rank'] == 1])
    success_counts.append(pref_success)
    
    # Standard methods
    for method in ['wachter', 'growing_spheres', 'prototype', 'gradient_based']:
        method_rows = df_std_valid[df_std_valid['method'] == method]
        method_success = len(method_rows[['sample_idx', 'target_idx']].drop_duplicates())
        success_counts.append(method_success)
    
    success_rates = [100 * count / total_pairs for count in success_counts]
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
    bars = ax.bar(methods, success_rates, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Counterfactual Generation Success Rate by Method', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/success_rate_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir}/success_rate_comparison.png")
    plt.close()
    
    # 2. L2 Distance Comparison (only for valid CFs)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    l2_data = []
    labels = []
    
    # Preference method
    l2_data.append(df_pref_best['cf_l2_distance'].values)
    labels.append('Preference')
    
    # Standard methods
    for method in ['wachter', 'growing_spheres', 'prototype', 'gradient_based']:
        method_rows = df_std_valid[df_std_valid['method'] == method]
        if len(method_rows) > 0:
            l2_data.append(method_rows['l2_distance'].values)
            labels.append(method.replace('_', ' ').title())
    
    bp = ax.boxplot(l2_data, labels=labels, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors[:len(l2_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('L2 Distance', fontsize=12, fontweight='bold')
    ax.set_title('L2 Distance Distribution by Method', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/l2_distance_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir}/l2_distance_comparison.png")
    plt.close()
    
    # 3. Computation Time Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_means = []
    time_stds = []
    
    # Preference method (unique times per pair)
    pref_times = df_pref_best.groupby(['sample_idx', 'target_idx'])['computation_time'].first()
    time_means.append(pref_times.mean())
    time_stds.append(pref_times.std())
    
    # Standard methods - time is for ALL methods combined per pair
    std_times = df_std_valid.groupby(['sample_idx', 'target_idx'])['computation_time'].first()
    time_means.append(std_times.mean())
    time_stds.append(std_times.std())
    
    method_labels = ['Preference-based', 'Standard Methods\n(All 4 combined)']
    colors_time = ['#2ecc71', '#e74c3c']
    
    bars = ax.bar(method_labels, time_means, yerr=time_stds, color=colors_time, 
                  alpha=0.7, edgecolor='black', capsize=10)
    
    # Add value labels
    for bar, mean_val in zip(bars, time_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.3f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Average Computation Time by Method', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/computation_time_comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_dir}/computation_time_comparison.png")
    plt.close()
    
    # 4. Preference Score Comparison (if available)
    if 'preference_score' in df_std_valid.columns and df_std_valid['preference_score'].notna().any():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        pref_score_data = []
        pref_labels = []
        
        # Preference method
        pref_score_data.append(df_pref_best['cf_preference_score'].values)
        pref_labels.append('Preference')
        
        # Standard methods
        for method in ['wachter', 'growing_spheres', 'prototype', 'gradient_based']:
            method_rows = df_std_valid[df_std_valid['method'] == method]
            valid_scores = method_rows[method_rows['preference_score'].notna()]['preference_score'].values
            
            if len(valid_scores) > 0:
                pref_score_data.append(valid_scores)
                pref_labels.append(method.replace('_', ' ').title())
        
        if len(pref_score_data) > 1:  # Only create plot if we have data
            bp = ax.boxplot(pref_score_data, labels=pref_labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors[:len(pref_score_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Preference Score', fontsize=12, fontweight='bold')
            ax.set_title('Preference Score Distribution by Method', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/preference_score_comparison.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved: {output_dir}/preference_score_comparison.png")
            plt.close()
    
    logger.info(f"All visualizations saved to {output_dir}/")


def main():
    """Main comparison analysis."""
    logger.info("\n" + "=" * 100)
    logger.info("STARTING COMPREHENSIVE METHOD COMPARISON")
    logger.info("=" * 100 + "\n")
    
    # Load results
    df_pref = load_preference_results()
    df_std = load_standard_results()
    
    # Get best CFs from preference method
    df_pref_best = get_best_cf_per_pair_preference(df_pref)
    
    # Get valid CFs from standard methods
    df_std_valid = get_valid_cfs_standard(df_std)
    
    # Load training data for preference score calculation
    logger.info("\nLoading training data for preference score calculation...")
    experiment = AutoMPGExperiment()
    X_train, X_test, y_train, y_test, scaler, model, X_full, y_full = experiment.load_data()
    
    # Calculate preference scores for standard methods using exemplars from preference results
    logger.info("\nCalculating preference scores for standard methods using actual exemplars...")
    exemplar_weight = df_pref['exemplar_weight'].iloc[0] if 'exemplar_weight' in df_pref.columns else 0.01
    df_std_valid = add_preference_scores_to_standard_methods(df_std_valid, df_pref, X_train, exemplar_weight)
    
    # Create focused score comparison table (NEW)
    df_scores = create_score_comparison_table(df_pref_best, df_std_valid)
    
    # Create comparison table
    df_comparison = create_comparison_table(df_pref_best, df_std_valid)
    
    # Save comparison table
    df_comparison.to_csv('method_comparison_table.csv', index=False)
    logger.info("\nComparison table saved to: method_comparison_table.csv")
    
    # Create summary statistics
    create_summary_statistics(df_pref_best, df_std_valid)
    
    # Create visualizations
    create_visualizations(df_pref_best, df_std_valid)
    
    logger.info("\n" + "=" * 100)
    logger.info("COMPARISON ANALYSIS COMPLETE")
    logger.info("=" * 100)
    logger.info("\nGenerated files:")
    logger.info("  - score_comparison_table.csv (NEW: Focused score comparison)")
    logger.info("  - method_comparison_table.csv")
    logger.info("  - comparison_plots/success_rate_comparison.png")
    logger.info("  - comparison_plots/l2_distance_comparison.png")
    logger.info("  - comparison_plots/computation_time_comparison.png")


if __name__ == "__main__":
    main()
