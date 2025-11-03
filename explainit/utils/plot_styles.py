"""
Plot styling utilities for explainit package.
Provides consistent styling across all visualization functions.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Color palette
COLORS = {
    'deep_teal': '#008080',
    'forest_green': '#228b22',
    'moss_green': '#8a9a5b',
    'warm_brown': '#8b4513',
    'steel_gray': '#46505a',
    'dusty_blue': '#6b8e9b',
    'dirty_white': '#f0f0eb',
    'charcoal': '#2d2d2d',
    'light_gray': '#b4b4b4',
    'dark_background': '#0f0f0f',
    'black_background': '#000000',
    'plot_background': '#191919',
    # 'deep_teal': (0/255, 128/255, 128/255),
    # 'forest_green': (34/255, 139/255, 34/255),
    # 'moss_green': (138/255, 154/255, 91/255),
    # 'warm_brown': (139/255, 69/255, 19/255),
    # 'steel_gray': (70/255, 80/255, 90/255),        
    # 'dusty_blue': (107/255, 142/255, 155/255),
    # 'dirty_white': (240/255, 240/255, 235/255),    # Main text color
    # 'charcoal': (45/255, 45/255, 45/255),          
    # 'light_gray': (180/255, 180/255, 180/255),     
    # 'dark_background': (15/255, 15/255, 15/255),   # Very dark background
    # 'black_background': (0/255, 0/255, 0/255),     # Pure black
    # 'plot_background': (25/255, 25/255, 25/255)    # Slightly lighter for plot area
}

# Color palette as list for easy access
COLOR_PALETTE = [
    COLORS['deep_teal'],
    COLORS['forest_green'], 
    COLORS['moss_green'],
    COLORS['warm_brown'],
    COLORS['steel_gray'],
    COLORS['dusty_blue']
]

def apply_style():
    """Apply consistent styling to matplotlib plots."""
    # Font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Bebas Neue', 'Luckiest Guy', 'Rockwell']
    
    # Font sizes - significantly larger
    plt.rcParams['font.size'] = 16           # Base font size increased
    plt.rcParams['axes.titlesize'] = 24      # Title size increased
    plt.rcParams['axes.labelsize'] = 20      # Axis label size increased  
    plt.rcParams['xtick.labelsize'] = 16     # X-tick label size increased
    plt.rcParams['ytick.labelsize'] = 16     # Y-tick label size increased
    plt.rcParams['legend.fontsize'] = 18     # Legend size increased
    
    # Dark theme colors with dirty white text
    plt.rcParams['axes.edgecolor'] = COLORS['dirty_white']
    plt.rcParams['axes.linewidth'] = 2.0
    plt.rcParams['grid.color'] = COLORS['steel_gray']
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['text.color'] = COLORS['dirty_white']
    plt.rcParams['axes.labelcolor'] = COLORS['dirty_white']
    plt.rcParams['xtick.color'] = COLORS['dirty_white']
    plt.rcParams['ytick.color'] = COLORS['dirty_white']
    
    # Dark backgrounds
    plt.rcParams['figure.facecolor'] = COLORS['dark_background']
    plt.rcParams['axes.facecolor'] = COLORS['plot_background']

def style_categorical_plot(ax, num_categories=1):
    """Apply dark theme styling to categorical bar plots with enhanced visual effects."""
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Style remaining spines with dirty white
    ax.spines['left'].set_color(COLORS['dirty_white'])
    ax.spines['bottom'].set_color(COLORS['dirty_white'])
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    
    # Enhanced grid with subtle contrast on dark background
    ax.grid(axis='y', alpha=0.4, color=COLORS['steel_gray'], linestyle='--', linewidth=1.2)
    ax.set_axisbelow(True)
    
    # Set dark background
    ax.set_facecolor(COLORS['plot_background'])
    
    # Title and labels with dirty white text and larger fonts
    ax.set_title(ax.get_title(), color=COLORS['dirty_white'], fontweight='bold', 
                fontsize=26, pad=30)
    ax.set_xlabel(ax.get_xlabel(), color=COLORS['dirty_white'], fontweight='bold', fontsize=22)
    ax.set_ylabel(ax.get_ylabel(), color=COLORS['dirty_white'], fontweight='bold', fontsize=22)
    
    # Tick colors with dirty white
    ax.tick_params(axis='x', colors=COLORS['dirty_white'], labelsize=18)
    ax.tick_params(axis='y', colors=COLORS['dirty_white'], labelsize=18)

def style_numerical_plot(ax, plot_type='line'):
    """Apply dark theme styling to numerical distribution plots with enhanced contrast."""
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Style remaining spines with dirty white
    ax.spines['left'].set_color(COLORS['dirty_white'])
    ax.spines['bottom'].set_color(COLORS['dirty_white'])
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    
    # Enhanced grid with good contrast on dark background
    ax.grid(True, alpha=0.4, color=COLORS['steel_gray'], linestyle='--', linewidth=1.2)
    ax.set_axisbelow(True)
    
    # Set dark background
    ax.set_facecolor(COLORS['plot_background'])
    
    # Title and labels with dirty white text and larger fonts
    ax.set_title(ax.get_title(), color=COLORS['dirty_white'], fontweight='bold', 
                fontsize=26, pad=30)
    ax.set_xlabel(ax.get_xlabel(), color=COLORS['dirty_white'], fontweight='bold', fontsize=22)
    ax.set_ylabel(ax.get_ylabel(), color=COLORS['dirty_white'], fontweight='bold', fontsize=22)
    
    # Tick colors with dirty white
    ax.tick_params(axis='x', colors=COLORS['dirty_white'], labelsize=18)
    ax.tick_params(axis='y', colors=COLORS['dirty_white'], labelsize=18)

def get_bar_color(index=0):
    """Get bar color for categorical plots with gradient and alpha effects."""
    base_color = COLOR_PALETTE[index % len(COLOR_PALETTE)]
    return base_color

def get_bar_gradient_colors(num_bars):
    """Get gradient colors for multiple bars with alpha variations."""
    colors = []
    for i in range(num_bars):
        base_color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        # Add slight alpha variation for depth
        alpha = 0.8 + 0.2 * (i % 2)  # Alternating alpha for visual interest
        colors.append((*base_color, alpha))
    return colors

def get_line_color(plot_type='theoretical'):
    """Get line colors for numerical distribution plots with enhanced contrast."""
    if plot_type == 'theoretical':
        return COLORS['deep_teal']
    elif plot_type == 'empirical':
        return COLORS['forest_green']
    else:
        return COLORS['warm_brown']

def create_gradient_background(ax, color1=None, color2=None, alpha=0.1):
    """Create a subtle gradient background for plots."""
    if color1 is None:
        color1 = COLORS['deep_teal']
    if color2 is None:
        color2 = COLORS['forest_green']
    
    # Create gradient
    gradient = LinearSegmentedColormap.from_list('custom_gradient', [color1, color2])
    
    # Apply gradient background
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    ax.imshow(Z, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], 
             cmap=gradient, alpha=alpha, aspect='auto')