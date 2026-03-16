"""
Template for High-Impact Scientific Data Visualization
using Python, matplotlib, and scienceplots.
"""

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

# 1. Environment Setup
plt.style.use(['science', 'ieee'])
# Optional: if you need grid
# plt.rcParams.update({'axes.grid': True})

def create_publication_plot(x, y1, y2, output_filename='my_figure.pdf', columns='double'):
    """
    Creates a publication-quality plot and saves it to a PDF file.

    Parameters:
        x (array-like): X-axis data.
        y1 (array-like): First dataset for Y-axis.
        y2 (array-like): Second dataset for Y-axis.
        output_filename (str): Name of the file to save the plot.
        columns (str): 'single' (3.3 inches) or 'double' (7.0 inches).
    """

    # 2. Defining Figure Dimensions
    if columns == 'single':
        width = 3.3
    else:
        width = 7.0
        
    # Standard aspect ratio (side-by-side or standard plot)
    # Using golden ratio (0.618) or 0.5 for wider plots
    height = width * 0.5
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(width, height))

    # 3. Formatting Rules
    # Plotting data (example uses different colormaps or line styles automatically via scienceplots)
    ax.plot(x, y1, label='Model Output')
    ax.plot(x, y2, linestyle='--', label='Observed Data')

    # Labeling (always include units)
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Yield (t/ha)')

    # Legend placement
    ax.legend(loc='best', frameon=True)

    # 4. Final Export Checklist
    plt.tight_layout()
    
    # Save for LaTeX (vector output, minimum 300 DPI)
    plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Plot successfully saved to {output_filename}")

if __name__ == '__main__':
    # Generate some dummy data for the example
    x_data = np.linspace(0, 10, 100)
    y_model = np.sin(x_data)
    y_obs = np.sin(x_data) + np.random.normal(0, 0.1, 100)
    
    # Test the template
    create_publication_plot(x_data, y_model, y_obs, output_filename='example_figure.pdf', columns='single')
