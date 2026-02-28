import numpy as np
import pandas as pd
import itertools
import random
import time
import logging
from math import factorial
from typing import List, Dict, Tuple, Callable, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class CategoricalShapley:
    """
    Calculate Shapley values with multiple methods and generate comparison reports.
    
    Methods:
    - Exact calculation without categorical grouping
    - Approximate calculation without categorical grouping
    - Exact calculation with categorical grouping
    - Approximate calculation with categorical grouping
    """
    
    def __init__(self, model_pred: Callable, categorical_groups: List[List[int]] = None):
        """
        Args:
            model_pred: Prediction function that takes array of samples
            categorical_groups: List of lists, each containing indices of one-hot encoded features
                              e.g., [[3,4,5], [6,7]] means features 3,4,5 form one group
        """
        self.model_pred = model_pred
        self.categorical_groups = categorical_groups or []
        self.results = {}
        
    def calculate_all(self, samples: List[np.ndarray], base_values: List[np.ndarray], 
                     num_samples: int = 200, use_exact: bool = False) -> Dict[str, Any]:
        """
        Calculate Shapley values using approximation methods (and optionally exact methods).
        
        Args:
            samples: List of sample vectors to explain
            base_values: List of base/reference vectors for each sample
            num_samples: Number of samples for approximation methods
            use_exact: If True, also calculate exact methods (WARNING: very slow for many features!)
            
        Returns:
            Dictionary with results from all methods
        """
        logger.info(f"Starting Shapley calculation for {len(samples)} sample(s)")
        logger.info(f"Approximation samples: {num_samples}")
        logger.info(f"Exact methods: {'ENABLED' if use_exact else 'DISABLED (use use_exact=True to enable)'}")
        if self.categorical_groups:
            logger.info(f"Categorical groups: {self.categorical_groups}")
        else:
            logger.info("No categorical grouping specified")
        
        self.results = {
            'samples': samples,
            'base_values': base_values,
            'num_samples': num_samples,
            'methods': {}
        }
        
        for idx, (sample, base) in enumerate(zip(samples, base_values)):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing sample {idx+1}/{len(samples)}")
            logger.info(f"Sample shape: {sample.shape}, Features: {len(sample)}")
            logger.info(f"{'='*60}")
            
            method_results = {}
            
            if use_exact:
                # Exact without grouping
                logger.info("[1/4] Calculating exact Shapley without grouping...")
                n_features = len(sample)
                n_subsets = n_features * (2 ** (n_features - 1))
                logger.warning(f"This will compute ~{n_subsets:,} subsets. This may take a VERY long time!")
                t0 = time.time()
                exact_no_group = self._exact_shapley(sample, base, use_grouping=False)
                t1 = time.time()
                logger.info(f"✓ Completed in {t1-t0:.4f}s")
                method_results['exact_no_group'] = {'values': exact_no_group, 'time': t1-t0}
            
            # Approximate without grouping
            method_num = "[1/2]" if not use_exact else "[2/4]"
            logger.info(f"{method_num} Calculating approximate Shapley without grouping ({num_samples} samples)...")
            t2 = time.time()
            approx_no_group = self._approx_shapley(sample, base, num_samples, use_grouping=False)
            t3 = time.time()
            logger.info(f"✓ Completed in {t3-t2:.4f}s")
            method_results['approx_no_group'] = {'values': approx_no_group, 'time': t3-t2}
            
            if use_exact:
                # Exact with grouping
                logger.info("[3/4] Calculating exact Shapley with grouping...")
                if self.categorical_groups:
                    n_groups = len(self.categorical_groups) + (len(sample) - sum(len(g) for g in self.categorical_groups))
                    n_subsets = n_groups * (2 ** (n_groups - 1))
                    logger.info(f"Computing ~{n_subsets:,} subsets...")
                t4 = time.time()
                exact_group = self._exact_shapley(sample, base, use_grouping=True)
                t5 = time.time()
                logger.info(f"✓ Completed in {t5-t4:.4f}s")
                method_results['exact_group'] = {'values': exact_group, 'time': t5-t4}
            
            # Approximate with grouping
            method_num = "[2/2]" if not use_exact else "[4/4]"
            logger.info(f"{method_num} Calculating approximate Shapley with grouping ({num_samples} samples)...")
            t6 = time.time()
            approx_group = self._approx_shapley(sample, base, num_samples, use_grouping=True)
            t7 = time.time()
            logger.info(f"✓ Completed in {t7-t6:.4f}s")
            method_results['approx_group'] = {'values': approx_group, 'time': t7-t6}
            
            self.results['methods'][idx] = method_results
            
            total_time = t7-t2 if not use_exact else t7-t0
            logger.info(f"Sample {idx+1} completed. Total time: {total_time:.4f}s")
        
        logger.info(f"\n{'='*60}")
        logger.info("All calculations completed successfully!")
        logger.info(f"{'='*60}\n")
        
        return self.results
    
    def _exact_shapley(self, sample: np.ndarray, base: np.ndarray, use_grouping: bool) -> np.ndarray:
        """Exact Shapley calculation"""
        if use_grouping and self.categorical_groups:
            return self._exact_grouped(sample, base)
        return self._exact_ungrouped(sample, base)
    
    def _approx_shapley(self, sample: np.ndarray, base: np.ndarray, 
                       num_samples: int, use_grouping: bool) -> np.ndarray:
        """Approximate Shapley calculation"""
        if use_grouping and self.categorical_groups:
            return self._approx_grouped(sample, base, num_samples)
        return self._approx_ungrouped(sample, base, num_samples)
    
    def _exact_ungrouped(self, sample: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Exact Shapley without grouping"""
        n = len(sample)
        phi = np.zeros(n)
        total_subsets = 2**n
        
        logger.debug(f"Computing exact Shapley for {n} features ({total_subsets} total subsets)")
        
        for i in range(n):
            if i % 5 == 0 and i > 0:
                logger.debug(f"  Feature {i}/{n}...")
            
            others = [j for j in range(n) if j != i]
            
            for k in range(len(others) + 1):
                for S in itertools.combinations(others, k):
                    S = set(S)
                    weight = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                    
                    z_without = base.copy()
                    z_without[list(S)] = sample[list(S)]
                    
                    z_with = z_without.copy()
                    z_with[i] = sample[i]
                    
                    pred_without = self.model_pred([z_without])[0]
                    pred_with = self.model_pred([z_with])[0]
                    
                    phi[i] += weight * (pred_with - pred_without)
        
        return phi
    
    def _approx_ungrouped(self, sample: np.ndarray, base: np.ndarray, num_samples: int) -> np.ndarray:
        """Approximate Shapley without grouping"""
        n = len(sample)
        phi = np.zeros(n)
        
        logger.debug(f"Computing approximate Shapley for {n} features using {num_samples} samples per feature")
        
        for i in range(n):
            if i % 5 == 0 and i > 0:
                logger.debug(f"  Feature {i}/{n}...")
            
            others = [j for j in range(n) if j != i]
            contributions = []
            
            for _ in range(num_samples):
                subset_size = random.randint(0, len(others))
                S = set(random.sample(others, subset_size)) if subset_size > 0 else set()
                
                z_without = base.copy()
                z_without[list(S)] = sample[list(S)]
                
                z_with = z_without.copy()
                z_with[i] = sample[i]
                
                pred_without = self.model_pred([z_without])[0]
                pred_with = self.model_pred([z_with])[0]
                
                contributions.append(pred_with - pred_without)
            
            phi[i] = np.mean(contributions)
        
        return phi
    
    def _exact_grouped(self, sample: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Exact Shapley with categorical grouping"""
        # Create feature mapping
        feature_map, n_groups = self._create_feature_mapping(len(sample))
        consolidated_sample, consolidated_base = self._consolidate_vectors(sample, base, feature_map)
        
        total_subsets = 2**n_groups
        logger.debug(f"Computing exact Shapley for {n_groups} groups ({total_subsets} total subsets)")
        logger.debug(f"Original {len(sample)} features consolidated into {n_groups} groups")
        
        phi_consolidated = np.zeros(n_groups)
        
        for i in range(n_groups):
            if i % 3 == 0 and i > 0:
                logger.debug(f"  Group {i}/{n_groups}...")
            
            others = [j for j in range(n_groups) if j != i]
            
            for k in range(len(others) + 1):
                for S in itertools.combinations(others, k):
                    S = set(S)
                    weight = factorial(len(S)) * factorial(n_groups - len(S) - 1) / factorial(n_groups)
                    
                    z_without = self._create_hybrid(consolidated_base, consolidated_sample, S, feature_map, len(sample))
                    z_with = self._create_hybrid(consolidated_base, consolidated_sample, S | {i}, feature_map, len(sample))
                    
                    pred_without = self.model_pred([z_without])[0]
                    pred_with = self.model_pred([z_with])[0]
                    
                    phi_consolidated[i] += weight * (pred_with - pred_without)
        
        return self._expand_to_original(phi_consolidated, feature_map, len(sample))
    
    def _approx_grouped(self, sample: np.ndarray, base: np.ndarray, num_samples: int) -> np.ndarray:
        """Approximate Shapley with categorical grouping"""
        # Create feature mapping
        feature_map, n_groups = self._create_feature_mapping(len(sample))
        consolidated_sample, consolidated_base = self._consolidate_vectors(sample, base, feature_map)
        
        logger.debug(f"Computing approximate Shapley for {n_groups} groups using {num_samples} samples per group")
        logger.debug(f"Original {len(sample)} features consolidated into {n_groups} groups")
        
        phi_consolidated = np.zeros(n_groups)
        
        for i in range(n_groups):
            if i % 3 == 0 and i > 0:
                logger.debug(f"  Group {i}/{n_groups}...")
            
            others = [j for j in range(n_groups) if j != i]
            contributions = []
            
            for _ in range(num_samples):
                subset_size = random.randint(0, len(others))
                S = set(random.sample(others, subset_size)) if subset_size > 0 else set()
                
                z_without = self._create_hybrid(consolidated_base, consolidated_sample, S, feature_map, len(sample))
                z_with = self._create_hybrid(consolidated_base, consolidated_sample, S | {i}, feature_map, len(sample))
                
                pred_without = self.model_pred([z_without])[0]
                pred_with = self.model_pred([z_with])[0]
                
                contributions.append(pred_with - pred_without)
            
            phi_consolidated[i] = np.mean(contributions)
        
        return self._expand_to_original(phi_consolidated, feature_map, len(sample))
    
    def _create_feature_mapping(self, n_features: int) -> Tuple[Dict, int]:
        """Map original features to groups"""
        feature_map = {}
        group_idx = 0
        
        # Non-grouped features
        grouped_indices = set(idx for group in self.categorical_groups for idx in group)
        for i in range(n_features):
            if i not in grouped_indices:
                feature_map[i] = group_idx
                group_idx += 1
        
        # Grouped features
        for group in self.categorical_groups:
            for idx in group:
                feature_map[idx] = group_idx
            group_idx += 1
        
        return feature_map, group_idx
    
    def _consolidate_vectors(self, sample: np.ndarray, base: np.ndarray, 
                            feature_map: Dict) -> Tuple[List, List]:
        """Convert to consolidated representation"""
        n_groups = max(feature_map.values()) + 1
        consolidated_sample = [None] * n_groups
        consolidated_base = [None] * n_groups
        
        for group in self.categorical_groups:
            group_idx = feature_map[group[0]]
            consolidated_sample[group_idx] = tuple(sample[i] for i in group)
            consolidated_base[group_idx] = tuple(base[i] for i in group)
        
        for i, group_idx in feature_map.items():
            if consolidated_sample[group_idx] is None:
                consolidated_sample[group_idx] = sample[i]
                consolidated_base[group_idx] = base[i]
        
        return consolidated_sample, consolidated_base
    
    def _create_hybrid(self, consolidated_base: List, consolidated_sample: List, 
                      S: set, feature_map: Dict, original_length: int) -> np.ndarray:
        """Create hybrid vector from consolidated representation"""
        z_consolidated = consolidated_base.copy()
        for i in S:
            z_consolidated[i] = consolidated_sample[i]
        
        # Expand to original format
        z = np.zeros(original_length)
        for orig_idx, group_idx in feature_map.items():
            val = z_consolidated[group_idx]
            if isinstance(val, tuple):
                # Find position in group
                group = [g for g in self.categorical_groups if orig_idx in g][0]
                pos = group.index(orig_idx)
                z[orig_idx] = val[pos]
            else:
                z[orig_idx] = val
        
        return z
    
    def _expand_to_original(self, phi_consolidated: np.ndarray, 
                           feature_map: Dict, original_length: int) -> np.ndarray:
        """Expand consolidated Shapley values to original feature space"""
        phi = np.zeros(original_length)
        for orig_idx, group_idx in feature_map.items():
            phi[orig_idx] = phi_consolidated[group_idx]
        return phi
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame for analysis.
        
        Returns:
            DataFrame with one row per calculation (sample × method combination).
            Each row contains lists of values for all features.
        """
        if not self.results:
            raise ValueError("No results available. Run calculate_all() first.")
        
        rows = []
        
        for sample_idx in self.results['methods'].keys():
            sample = self.results['samples'][sample_idx]
            base = self.results['base_values'][sample_idx]
            methods = self.results['methods'][sample_idx]
            
            # Determine feature types for all features
            grouped_indices = set(idx for group in self.categorical_groups for idx in group)
            feature_types = ['categorical' if i in grouped_indices else 'numerical' 
                           for i in range(len(sample))]
            
            # Map method names to standardized names
            method_name_map = {
                'exact_no_group': 'exact_wo_grouping',
                'approx_no_group': 'approx_wo_grouping',
                'exact_group': 'exact_w_grouping',
                'approx_group': 'approx_w_grouping'
            }
            
            # Calculate predictions for base and sample values
            base_pred = self.model_pred([base])[0]
            sample_pred = self.model_pred([sample])[0]
            
            for method_key, method_name in method_name_map.items():
                # Skip methods that weren't calculated
                if method_key not in methods:
                    continue
                    
                shap_values = methods[method_key]['values']
                time_taken = methods[method_key]['time']
                
                rows.append({
                    'sample_id': sample_idx,
                    'type': method_name,
                    'shap_values': list(shap_values),
                    'sample_values': list(sample),
                    'base_values': list(base),
                    'base_vals_pred': base_pred,
                    'sample_vals_pred': sample_pred,
                    'num_features': len(sample),
                    'feature_types': feature_types,
                    'num_categorical_groups': len(self.categorical_groups),
                    'approx_samples': self.results['num_samples'] if 'approx' in method_name else None,
                    'time_seconds': time_taken
                })
        
        df = pd.DataFrame(rows)
        return df
    
    def save_dataframe(self, filepath: str) -> pd.DataFrame:
        """
        Save results as CSV file and return the DataFrame.
        
        Args:
            filepath: Path to save the CSV file
            
        Returns:
            DataFrame with all results
        """
        logger.info(f"Saving results to {filepath}...")
        df = self.get_dataframe()
        df.to_csv(filepath, index=False)
        logger.info(f"✓ Results saved successfully ({df.shape[0]} rows, {df.shape[1]} columns)")
        return df
    
    def generate_report(self) -> str:
        """Generate comprehensive comparison report"""
        if not self.results:
            return "No results available. Run calculate_all() first."
        
        report = []
        report.append("=" * 80)
        report.append("SHAPLEY VALUES CALCULATION REPORT")
        report.append("=" * 80)
        report.append(f"\nNumber of samples: {len(self.results['samples'])}")
        report.append(f"Approximation samples: {self.results['num_samples']}")
        
        if self.categorical_groups:
            report.append(f"\nCategorical groups: {self.categorical_groups}")
        else:
            report.append("\nNo categorical grouping specified")
        
        for idx in self.results['methods'].keys():
            report.append(f"\n{'-'*80}")
            report.append(f"SAMPLE {idx+1}")
            report.append(f"{'-'*80}")
            
            sample = self.results['samples'][idx]
            base = self.results['base_values'][idx]
            report.append(f"\nSample shape: {sample.shape}")
            report.append(f"Number of features: {len(sample)}")
            
            methods = self.results['methods'][idx]
            
            # Timing comparison
            report.append("\n--- COMPUTATION TIME ---")
            for method_name, data in methods.items():
                report.append(f"{method_name:20s}: {data['time']:.4f} seconds")
            
            # Values comparison
            report.append("\n--- SHAPLEY VALUES ---")
            for method_name, data in methods.items():
                report.append(f"\n{method_name}:")
                values = data['values']
                for i, val in enumerate(values[:10]):  # Show first 10
                    report.append(f"  Feature {i}: {val:.6f}")
                if len(values) > 10:
                    report.append(f"  ... ({len(values)-10} more features)")
            
            # Error analysis (only if exact methods were calculated)
            if 'exact_no_group' in methods:
                report.append("\n--- ERROR ANALYSIS ---")
                exact_ng = methods['exact_no_group']['values']
                
                if 'approx_no_group' in methods:
                    approx_ng = methods['approx_no_group']['values']
                    mae_approx_ng = np.mean(np.abs(exact_ng - approx_ng))
                    report.append(f"MAE (Exact no group vs Approx no group): {mae_approx_ng:.6f}")
                
                if 'exact_group' in methods:
                    exact_g = methods['exact_group']['values']
                    mae_exact_g = np.mean(np.abs(exact_ng - exact_g))
                    report.append(f"MAE (Exact no group vs Exact group):     {mae_exact_g:.6f}")
                
                if 'approx_group' in methods:
                    approx_g = methods['approx_group']['values']
                    mae_approx_g = np.mean(np.abs(exact_ng - approx_g))
                    report.append(f"MAE (Exact no group vs Approx group):    {mae_approx_g:.6f}")
                
                # Speedup
                if 'approx_no_group' in methods:
                    speedup_approx = methods['exact_no_group']['time'] / methods['approx_no_group']['time']
                    report.append(f"\nSpeedup (Approximation):     {speedup_approx:.2f}x")
                if 'exact_group' in methods and self.categorical_groups:
                    speedup_group = methods['exact_no_group']['time'] / methods['exact_group']['time']
                    report.append(f"Speedup (Grouping):          {speedup_group:.2f}x")
            elif 'approx_no_group' in methods and 'approx_w_grouping' in methods:
                # Compare two approximation methods
                report.append("\n--- COMPARISON ---")
                approx_ng = methods['approx_no_group']['values']
                approx_g = methods['approx_group']['values']
                mae_approx = np.mean(np.abs(approx_ng - approx_g))
                report.append(f"MAE (Approx no group vs Approx group): {mae_approx:.6f}")
                report.append("\nNote: Exact methods were not calculated (use use_exact=True to enable)")
        
        report.append(f"\n{'='*80}\n")
        return "\n".join(report)
