"""
Configuration Examples for Counterfactual Experiments

This file shows various configuration options for running experiments.
Copy the configuration you want to use into experiment_final.py or test_experiment.py
"""

# ============================================================================
# EXAMPLE 1: Run both methods (default)
# ============================================================================
config_both_methods = {
    # Dataset and model
    'dataset': 'Auto_MPG',
    'model': 'NN_Residual',
    
    # Experiment settings
    'n_quantiles': 3,
    'epsilon': 2.0,
    
    # Method selection
    'run_preference_method': True,
    'run_standard_methods': True,
    
    # Standard methods (all 4 available)
    'standard_methods': ['wachter', 'growing_spheres', 'prototype', 'gradient_based'],
    
    # Preference method settings
    'return_top_n': 5,
    'exemplar_weight': 0.01,
    'n_samples': 10000,
    'use_monte_carlo': True,
}


# ============================================================================
# EXAMPLE 2: Run only your preference-based method
# ============================================================================
config_preference_only = {
    'dataset': 'Auto_MPG',
    'model': 'NN_Residual',
    
    'n_quantiles': 3,
    'epsilon': 2.0,
    
    # Enable only preference method
    'run_preference_method': True,
    'run_standard_methods': False,  # Skip standard methods
    
    'standard_methods': [],  # Not used since run_standard_methods=False
    
    # Preference method settings
    'return_top_n': 5,
    'exemplar_weight': 0.01,
    'n_samples': 10000,
    'use_monte_carlo': True,
}


# ============================================================================
# EXAMPLE 3: Run only standard methods (no preference method)
# ============================================================================
config_standard_only = {
    'dataset': 'Auto_MPG',
    'model': 'NN_Residual',
    
    'n_quantiles': 3,
    'epsilon': 2.0,
    
    # Enable only standard methods
    'run_preference_method': False,  # Skip preference method
    'run_standard_methods': True,
    
    # Run all 4 standard methods
    'standard_methods': ['wachter', 'growing_spheres', 'prototype', 'gradient_based'],
    
    # These are not used since run_preference_method=False
    'return_top_n': 5,
    'exemplar_weight': 0.01,
    'n_samples': 10000,
    'use_monte_carlo': True,
}


# ============================================================================
# EXAMPLE 4: Run only specific standard methods
# ============================================================================
config_selected_standard_methods = {
    'dataset': 'Auto_MPG',
    'model': 'NN_Residual',
    
    'n_quantiles': 3,
    'epsilon': 2.0,
    
    'run_preference_method': False,
    'run_standard_methods': True,
    
    # Only run Wachter and Growing Spheres (skip prototype and gradient-based)
    'standard_methods': ['wachter', 'growing_spheres'],
    
    'return_top_n': 5,
    'exemplar_weight': 0.01,
    'n_samples': 10000,
    'use_monte_carlo': True,
}


# ============================================================================
# EXAMPLE 5: Quick test configuration (fast execution)
# ============================================================================
config_quick_test = {
    'dataset': 'Auto_MPG',
    'model': 'NN_Residual',
    
    # Minimal experiments
    'n_quantiles': 2,  # Only 2 sample points = 2 experiments
    'epsilon': 3.0,    # Larger tolerance for faster success
    
    'run_preference_method': True,
    'run_standard_methods': True,
    
    # Only fast methods
    'standard_methods': ['wachter', 'growing_spheres'],  # Skip slower methods
    
    # Quick preference settings
    'return_top_n': 3,
    'exemplar_weight': 0.01,
    'n_samples': 1000,  # Fewer samples for speed
    'use_monte_carlo': True,
}


# ============================================================================
# EXAMPLE 6: Comprehensive comparison (slower, thorough)
# ============================================================================
config_comprehensive = {
    'dataset': 'Auto_MPG',
    'model': 'NN_Residual',
    
    # More sample points
    'n_quantiles': 5,  # 5 points = 20 experiments
    'epsilon': 2.0,    # Stricter tolerance
    
    'run_preference_method': True,
    'run_standard_methods': True,
    
    # All standard methods
    'standard_methods': ['wachter', 'growing_spheres', 'prototype', 'gradient_based'],
    
    # High-quality preference settings
    'return_top_n': 10,      # Keep top 10 CFs
    'exemplar_weight': 0.01,
    'n_samples': 50000,      # Many samples for better results
    'use_monte_carlo': True,
}


# ============================================================================
# EXAMPLE 7: Compare only gradient-based methods
# ============================================================================
config_gradient_comparison = {
    'dataset': 'Auto_MPG',
    'model': 'NN_Residual',
    
    'n_quantiles': 3,
    'epsilon': 2.0,
    
    'run_preference_method': True,  # Your method uses preferences
    'run_standard_methods': True,
    
    # Only gradient-based method from standard methods
    'standard_methods': ['gradient_based'],
    
    'return_top_n': 5,
    'exemplar_weight': 0.01,
    'n_samples': 10000,
    'use_monte_carlo': True,
}


# ============================================================================
# EXAMPLE 8: Instance-based methods only (prototype + growing spheres)
# ============================================================================
config_instance_based = {
    'dataset': 'Auto_MPG',
    'model': 'NN_Residual',
    
    'n_quantiles': 3,
    'epsilon': 2.0,
    
    'run_preference_method': False,
    'run_standard_methods': True,
    
    # Methods that use training instances
    'standard_methods': ['growing_spheres', 'prototype'],
    
    'return_top_n': 5,
    'exemplar_weight': 0.01,
    'n_samples': 10000,
    'use_monte_carlo': True,
}


# ============================================================================
# HOW TO USE
# ============================================================================
"""
To use one of these configurations:

1. Open experiment_final.py
2. Find the 'config' dictionary in the main() function
3. Replace it with one of the configurations above

Example:
    def main():
        # Replace this section:
        config = {
            'dataset': 'Auto_MPG',
            ...
        }
        
        # With:
        from config_examples import config_quick_test
        config = config_quick_test

Or simply copy-paste the dictionary values you want to change.
"""


# ============================================================================
# AVAILABLE OPTIONS REFERENCE
# ============================================================================
"""
DATASET OPTIONS:
    'dataset': 'Auto_MPG'  # Currently only Auto_MPG supported

MODEL OPTIONS:
    'model': 'NN_Residual'  # Currently only NN_Residual supported

EXPERIMENT SETTINGS:
    'n_quantiles': int  # Number of quantile points (2-10 recommended)
                        # n_quantiles=2 -> 2 experiments
                        # n_quantiles=3 -> 6 experiments  
                        # n_quantiles=5 -> 20 experiments
    
    'epsilon': float    # Target tolerance for predictions
                        # Smaller = stricter (harder to find CFs)
                        # Larger = looser (easier to find CFs)
                        # Recommended: 2.0-5.0 for MPG

METHOD SELECTION:
    'run_preference_method': bool  # True = run your preference-based method
                                   # False = skip it
    
    'run_standard_methods': bool   # True = run standard methods
                                   # False = skip them
    
    'standard_methods': list       # Which standard methods to run
                                   # Options: 'wachter', 'growing_spheres', 
                                   #          'prototype', 'gradient_based'
                                   # Example: ['wachter', 'prototype']
                                   # Empty list [] = none

PREFERENCE METHOD SETTINGS:
    'return_top_n': int         # How many top CFs to save per experiment
                                # Recommended: 3-10
    
    'exemplar_weight': float    # Weight for exemplar in preferences
                                # Recommended: 0.01-0.5
    
    'n_samples': int            # Number of candidates to generate
                                # More = better quality, slower
                                # Recommended: 1000 (fast), 10000 (default), 50000 (thorough)
    
    'use_monte_carlo': bool     # Use Monte Carlo sampling
                                # Recommended: True

STANDARD METHOD DETAILS:
    'wachter': Optimization-based, finds CFs via gradient descent
               - Good balance of speed and quality
               - Works well for most cases
    
    'growing_spheres': Geometric search, expands from original point
                       - Fast
                       - Uses dataset structure
    
    'prototype': Uses real instances from training data
                 - Most trustworthy (real data points)
                 - May not be closest
    
    'gradient_based': Direct gradient descent on neural network
                      - Very fast
                      - Requires TensorFlow model
                      - May produce larger changes
"""
