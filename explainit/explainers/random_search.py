
from explainit.logging_config import logger
from explainit.utils.plot_styles import (apply_style, style_numerical_plot, style_categorical_plot, 
                                        get_line_color, get_bar_color, get_bar_gradient_colors, COLORS)
# logger.info("This is an info message")
# logger.debug("This is a debug message with details")
# logger.warning("This is a warning message")
# logger.error("This is an error message")

import numpy as np
import random
import matplotlib.pyplot as plt


class RandomSearchExplainer:
    def __init__(self, model_pred, priorities, sample, target,):
        self.model_pred = model_pred
        self.priorities = priorities
        self.sample = sample
        self.target = target
        self.filtered_priorities = self.filter_priorities()

    def filter_priorities(self):
        """
        Filters the priorities dictionary by removing unactionable features.
        Unactionable numerical features (with priority 0) are replaced with the sample value.
        """
        
        new_priorities = {'numerical': {}, 'categorical': {}}

        # Numerical: replace 0 with sample value
        for idx, val in self.priorities['numerical'].items():
            if val == 0:
                new_priorities['numerical'][idx] = self.sample[idx]
            else:
                new_priorities['numerical'][idx] = val

        # Categorical: remove elements with value 0
        for group, mapping in self.priorities['categorical'].items():
            filtered_mapping = {k: v for k, v in mapping.items() if v != 0}
            if filtered_mapping:
                new_priorities['categorical'][group] = filtered_mapping

        return new_priorities

    def sample_numeric_value(self, constraint, max_tries=100):
        """Uses Monte Carlo sampling to sample a numeric value based on a constraint function.
        - Rejection Sampling (your primary method)
        - Inverse Transform Sampling (your fallback, discretized)"""

        min_val = constraint['min']
        max_val = constraint['max']
        f = constraint['function']

        # Rejection sampling (works with any shape as long as 0 <= f(x) <= 1)
        for _ in range(max_tries):
            rv = np.random.uniform(min_val, max_val)
            w = float(np.asarray(f(rv)).squeeze())
            if np.random.random() < w:
                return rv

        # Fallback: sample proportional to f over a grid (shape-agnostic)
        xs = np.linspace(min_val, max_val, 256)
        ws = np.asarray(f(xs)).astype(float).ravel()
        ws = np.clip(ws, 0.0, None)
        if ws.sum() == 0 or not np.isfinite(ws).all():
            return np.random.uniform(min_val, max_val)
        p = ws / ws.sum()
        idx = np.random.choice(len(xs), p=p)
        jitter = (max_val - min_val) / 256 * (np.random.random() - 0.5)
        return float(np.clip(xs[idx] + jitter, min_val, max_val))

    def investigate_probability_distribution(self, n_samples=10000):
        # Apply styling
        apply_style()

        for idx, constraint in self.priorities['numerical'].items():

            if isinstance(constraint, dict) and 'function' in constraint:
                min_val = constraint['min']
                max_val = constraint['max']
                f = constraint['function']
            else:
                print('Function for idx: ', idx, ' is unactionable (', constraint, ')')
                continue
            def calculate_empirical_distribution(n_samples=10000):
                accepted_values = []
                
                for _ in range(n_samples):
                    val = self.sample_numeric_value({
                        'function': f, 
                        'min': min_val, 
                        'max': max_val
                    })
                    accepted_values.append(val)
                
                return np.array(accepted_values)

            # Calculate theoretical distribution
            x_vals = np.linspace(min_val, max_val, 1000)
            weights = np.array([float(f(x)) for x in x_vals])

            # Normalize to get probability density
            prob_density = weights / np.trapezoid(weights, x_vals)

            # Generate samples
            samples = calculate_empirical_distribution(n_samples)

            # Calculate histogram (probability mass in bins)
            hist, bin_edges = np.histogram(samples, bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Create enhanced plot with gradient effects
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot theoretical distribution with enhanced styling
            ax.plot(x_vals, prob_density, label='Theoretical Distribution', 
                   color=get_line_color('theoretical'), linewidth=4, alpha=0.9,
                   solid_capstyle='round')
            
            # Add subtle fill under theoretical curve for depth
            ax.fill_between(x_vals, prob_density, alpha=0.2, 
                           color=get_line_color('theoretical'))
            
            # Plot empirical distribution with enhanced dark theme styling
            bar_colors = get_line_color('empirical')
            bars = ax.bar(bin_centers, hist, width=np.diff(bin_edges), alpha=0.8, 
                         label='Empirical Distribution', color=bar_colors,
                         edgecolor=COLORS['dirty_white'], linewidth=1.5)
            
            # Add gradient effect to bars
            for i, bar in enumerate(bars):
                # Alternate alpha for gradient effect
                bar.set_alpha(0.6 + 0.3 * (i % 2 == 0))
            
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Probability Density')
            ax.set_title(f'Probability Distribution for Numerical Feature {idx}')
            
            # Enhanced legend positioned to the right side of title, above plot
            legend = ax.legend(frameon=True, fancybox=True, shadow=True, 
                             facecolor=COLORS['dark_background'], edgecolor=COLORS['dirty_white'],
                             fontsize=16, loc='center left', bbox_to_anchor=(1.02, 0.9), ncol=1)
            legend.get_frame().set_alpha(0.9)
            # Make legend text dirty white
            for text in legend.get_texts():
                text.set_color(COLORS['dirty_white'])
            
            # Apply numerical plot styling
            style_numerical_plot(ax)
            
            # Adjust layout to accommodate legend to the right
            plt.tight_layout()
            plt.subplots_adjust(right=0.75)  # Make room for legend on the right
            plt.show()

        for group_indices, possible_values in self.priorities['categorical'].items():
            
            # Extract categories and their weights
            categories = list(possible_values.keys())
            weights = np.array(list(possible_values.values()), dtype=float)
            
            # Filter out forbidden combinations (weight = 0)
            allowed_mask = weights > 0
            
            if not np.any(allowed_mask):
                print(f'All combinations for categorical group {group_indices} are forbidden (weight = 0)')
                continue
            
            # Keep only allowed combinations
            allowed_categories = [categories[i] for i in range(len(categories)) if allowed_mask[i]]
            allowed_weights = weights[allowed_mask]
            
            # Normalize weights to probabilities
            probabilities = allowed_weights / allowed_weights.sum()
            
            # Create labels for the categories (convert tuples to strings for display)
            category_labels = [str(cat) if isinstance(cat, tuple) else str(cat) for cat in allowed_categories]
            
            # Calculate appropriate bar width based on number of categories
            num_categories = len(category_labels)
            print(num_categories)
            if num_categories == 1:
                bar_width = 0.1  # Very narrow for single category
            elif num_categories == 2:
                bar_width = 0.4  # Narrow for 2 categories
            elif num_categories <= 4:
                bar_width = 0.5  # Narrow for 3-4 categories
            elif num_categories <= 6:
                bar_width = 0.6  # Moderate for 5-6 categories
            else:
                bar_width = 0.8  # Standard width for many categories
            
            # Create bar plot with enhanced styling and gradients
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create gradient colors for bars with varying alpha
            gradient_colors = get_bar_gradient_colors(len(category_labels))
            bar_colors = [get_bar_color(i) for i in range(len(category_labels))]
            
            bars = ax.bar(range(len(category_labels)), probabilities, width=bar_width, 
                         color=bar_colors, edgecolor=COLORS['dirty_white'], linewidth=2.0)
            
            # Apply gradient alpha effect to each bar
            for i, bar in enumerate(bars):
                # Create depth with alternating alpha and subtle gradients
                alpha_val = 0.7 + 0.2 * (i % 2)
                bar.set_alpha(alpha_val)
                
                # Add subtle inner border for depth with dark theme
                height = bar.get_height()
                if height > 0:
                    ax.add_patch(plt.Rectangle((bar.get_x() + 0.02, 0.01), 
                                             bar.get_width() - 0.04, height - 0.02,
                                             fill=False, edgecolor=COLORS['steel_gray'], 
                                             linewidth=0.8, alpha=0.6))
            
            # For single category, adjust x-axis limits to center the bar better
            if num_categories == 1:
                ax.set_xlim(-1, 1)
            
            # Add value labels with enhanced dark theme styling
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, 
                       f'{prob:.3f}', ha='center', va='bottom', fontsize=16,
                       fontweight='bold', color=COLORS['dirty_white'],
                       bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['dark_background'], 
                                alpha=0.8, edgecolor=COLORS['dirty_white'], linewidth=1.5))
            
            ax.set_xlabel('Category Combinations')
            ax.set_ylabel('Probability')
            ax.set_title(f'Probability Distribution for Categorical Features {group_indices}')
            ax.set_xticks(range(len(category_labels)))
            ax.set_xticklabels(category_labels, rotation=45, ha='right')
            
            # Apply categorical plot styling
            style_categorical_plot(ax, num_categories)
            
            plt.tight_layout()
            plt.show()

    def display_priorities(self):
        """
        Display the priority functions and values for both numerical and categorical features.
        Similar to investigate_probability_distribution but shows the raw priority functions.
        """
        # Apply styling
        apply_style()

        # Display numerical feature priorities
        for idx, constraint in self.priorities['numerical'].items():

            if isinstance(constraint, dict) and 'function' in constraint:
                min_val = constraint['min']
                max_val = constraint['max']
                f = constraint['function']
                
                # Calculate priority function values
                x_vals = np.linspace(min_val, max_val, 1000)
                priority_values = np.array([float(f(x)) for x in x_vals])

                # Create enhanced plot with gradient effects
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Plot priority function with enhanced styling
                ax.plot(x_vals, priority_values, label='Priority Function', 
                       color=get_line_color('theoretical'), linewidth=4, alpha=0.9,
                       solid_capstyle='round')
                
                # Add subtle fill under priority curve for depth
                ax.fill_between(x_vals, priority_values, alpha=0.2, 
                               color=get_line_color('theoretical'))
                
                # Mark the current sample value
                sample_value = self.sample[idx]
                if min_val <= sample_value <= max_val:
                    sample_priority = float(f(sample_value))
                    ax.plot(sample_value, sample_priority, 'o', 
                           color=get_line_color('empirical'), markersize=12, 
                           label=f'Current Sample Value ({sample_value:.3f})',
                           markeredgecolor=COLORS['dirty_white'], markeredgewidth=2)
                
                ax.set_xlabel('Feature Value')
                ax.set_ylabel('Priority Weight')
                ax.set_title(f'Priority Function for Numerical Feature {idx}')
                
                # Enhanced legend positioned to the right side of title, above plot
                legend = ax.legend(frameon=True, fancybox=True, shadow=True, 
                                 facecolor=COLORS['dark_background'], edgecolor=COLORS['dirty_white'],
                                 fontsize=16, loc='center left', bbox_to_anchor=(1.02, 0.9), ncol=1)
                legend.get_frame().set_alpha(0.9)
                # Make legend text dirty white
                for text in legend.get_texts():
                    text.set_color(COLORS['dirty_white'])
                
                # Apply numerical plot styling
                style_numerical_plot(ax)
                
                # Adjust layout to accommodate legend to the right
                plt.tight_layout()
                plt.subplots_adjust(right=0.75)  # Make room for legend on the right
                plt.show()
            else:
                print(f'Feature {idx} is unactionable with fixed value: {constraint}')

        # Display categorical feature priorities
        for group_indices, possible_values in self.priorities['categorical'].items():
            
            # Extract categories and their weights
            categories = list(possible_values.keys())
            weights = np.array(list(possible_values.values()), dtype=float)
            
            # Create labels for the categories (convert tuples to strings for display)
            category_labels = [str(cat) if isinstance(cat, tuple) else str(cat) for cat in categories]
            
            # Calculate appropriate bar width based on number of categories
            num_categories = len(category_labels)
            if num_categories == 1:
                bar_width = 0.1  # Very narrow for single category
            elif num_categories == 2:
                bar_width = 0.4  # Narrow for 2 categories
            elif num_categories <= 4:
                bar_width = 0.5  # Narrow for 3-4 categories
            elif num_categories <= 6:
                bar_width = 0.6  # Moderate for 5-6 categories
            else:
                bar_width = 0.8  # Standard width for many categories
            
            # Create bar plot with enhanced styling and gradients
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create gradient colors for bars
            bar_colors = [get_bar_color(i) for i in range(len(category_labels))]
            
            bars = ax.bar(range(len(category_labels)), weights, width=bar_width, 
                         color=bar_colors, edgecolor=COLORS['dirty_white'], linewidth=2.0)
            
            # Apply gradient alpha effect to each bar
            for i, bar in enumerate(bars):
                # Create depth with alternating alpha and subtle gradients
                alpha_val = 0.7 + 0.2 * (i % 2)
                bar.set_alpha(alpha_val)
                
                # Add subtle inner border for depth with dark theme
                height = bar.get_height()
                if height > 0:
                    ax.add_patch(plt.Rectangle((bar.get_x() + 0.02, 0.01), 
                                             bar.get_width() - 0.04, height - 0.02,
                                             fill=False, edgecolor=COLORS['steel_gray'], 
                                             linewidth=0.8, alpha=0.6))
            
            # Highlight current sample values if they exist in the categories
            current_sample_combo = tuple(self.sample[idx] for idx in group_indices)
            if current_sample_combo in categories:
                current_idx = categories.index(current_sample_combo)
                bars[current_idx].set_edgecolor(get_line_color('empirical'))
                bars[current_idx].set_linewidth(4)
                # Add marker on top of the current sample bar
                ax.plot(current_idx, weights[current_idx] + max(weights) * 0.05, 'v', 
                       color=get_line_color('empirical'), markersize=15,
                       markeredgecolor=COLORS['dirty_white'], markeredgewidth=2)
            
            # For single category, adjust x-axis limits to center the bar better
            if num_categories == 1:
                ax.set_xlim(-1, 1)
            
            # Add value labels with enhanced dark theme styling
            for i, (bar, weight) in enumerate(zip(bars, weights)):
                label_text = f'{weight:.3f}'
                if i == categories.index(current_sample_combo) if current_sample_combo in categories else -1:
                    label_text += ' (Current)'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(weights) * 0.01, 
                       label_text, ha='center', va='bottom', fontsize=16,
                       fontweight='bold', color=COLORS['dirty_white'],
                       bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['dark_background'], 
                                alpha=0.8, edgecolor=COLORS['dirty_white'], linewidth=1.5))
            
            ax.set_xlabel('Category Combinations')
            ax.set_ylabel('Priority Weight')
            ax.set_title(f'Priority Weights for Categorical Features {group_indices}')
            ax.set_xticks(range(len(category_labels)))
            ax.set_xticklabels(category_labels, rotation=45, ha='right')
            
            # Apply categorical plot styling
            style_categorical_plot(ax, num_categories)
            
            plt.tight_layout()
            plt.show()


    def generate_random_samples(self, n_samples=1000, epsilon=0.05, random_seed=None, use_monte_carlo=True, max_tries=100):
        """
        Generate random samples based on filtered priorities.
        Only keep samples whose prediction is within epsilon of the target.
        n_samples: number of samples to generate
        epsilon: acceptable deviation from target prediction
        random_seed: seed for reproducibility
        use_monte_carlo: if True, use Monte Carlo sampling for numerical features; otherwise use uniform sampling
        max_tries: maximum number of tries for rejection sampling in numerical features (only if use_monte_carlo is True)
        Returns: list of samples and their corresponding predictions
        """

        # Only set seed if random_seed is provided
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        np.random.seed(random_seed)
        random.seed(random_seed)
        
        samples = []
        predictions = []
        n_features = 29
        
        for _ in range(n_samples):
            sample = np.zeros(n_features)
            # Numerical features
            for idx, constraint in self.filtered_priorities['numerical'].items():
                if isinstance(constraint, dict) and 'function' in constraint:
                    sample[idx] = self.sample_numeric_value(constraint, max_tries) if use_monte_carlo else np.random.uniform(constraint['min'], constraint['max'])
                else:
                    sample[idx] = constraint

            # Categorical features: sample combos with probability proportional to their weights
            for group_indices, possible_values in self.filtered_priorities['categorical'].items():
                combos = list(possible_values.keys())
                weights = np.array(list(possible_values.values()), dtype=float)
                
                # Filter out forbidden combinations (weight = 0)
                allowed_mask = weights > 0
                
                if not np.any(allowed_mask):
                    raise ValueError(f"All combinations for categorical group {group_indices} are forbidden (weight = 0). At least one combination must have weight > 0.")
                
                if use_monte_carlo:
                    # Monte Carlo sampling: sample according to probabilities (weights)
                    # Keep only allowed combinations
                    allowed_combos = [combos[i] for i in range(len(combos)) if allowed_mask[i]]
                    allowed_weights = weights[allowed_mask]

                    # Normalize weights to probabilities
                    allowed_weights = allowed_weights / allowed_weights.sum()
                    
                    # Sample from allowed combinations according to probabilities
                    sel = allowed_combos[np.random.choice(len(allowed_combos), p=allowed_weights)]
                else:
                    # Uniform sampling: sample uniformly from allowed combinations
                    allowed_combos = [combos[i] for i in range(len(combos)) if allowed_mask[i]]
                    sel = random.choice(allowed_combos)
                
                for i, idx in enumerate(group_indices):
                    sample[idx] = sel[i]

            # Get prediction
            try:
                pred = self.model_pred(sample.reshape(1, -1))[0]
                if abs(pred - self.target) <= epsilon:
                    samples.append(sample.copy())
                    predictions.append(pred)
            except Exception as e:
                logger.warning(f"Could not get prediction for sample: {e}")
        return samples, predictions

