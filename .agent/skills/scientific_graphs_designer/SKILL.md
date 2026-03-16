---
name: scientific_graphs_designer
description: Guide to creating high-impact publication-quality figures using Python, matplotlib, and scienceplots.
---

# SKILL: High-Impact Scientific Data Visualization

This guide outlines the standard workflow for creating publication-quality figures using Python. We utilize `matplotlib` and `scienceplots` to ensure compliance with strict journal standards (like IEEE or Nature).

## 1. Environment Setup

Always begin your script by importing the necessary libraries and setting the global style. The `science` and `ieee` styles automatically adjust fonts (often to Times New Roman), font sizes, and line widths to match professional standards.

```python
import matplotlib.pyplot as plt
import scienceplots

# Set the global style at the start
plt.style.use(['science', 'ieee'])
```

## 2. Defining Figure Dimensions

To avoid "surprises" in your LaTeX document, **never** let Matplotlib guess the figure size. You must define a width that matches your paper's column width and a height relative to that width.

### The Calculation

Standard IEEE column widths:

* **Single Column:** 3.3 inches.
* **Double Column (Full Width):** 7.0 inches.

By default we will use the double column width.

### Standard Aspect Ratios

* **Golden Ratio:** `height = width * 0.618`
* **Side-by-Side Subplots:** `height = width * 0.5` (Prevents vertical stretching)

```python
# Double-column example
width = 7.0 
height = width * 0.5 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height))
```

## 3. Formatting Rules for "Formal" Graphs

To maintain academic integrity and clarity, follow these rules:

* **Vector Output:** Always export to `.pdf`. This keeps text and lines as mathematical paths that never pixelate when zoomed.
* **Perceptual Color Maps:** Use `viridis`, `magma`, or `plasma`. Avoid `jet` or `rainbow` as they are not colorblind-friendly and do not print well in grayscale.
* **Labeling:** Every axis must have a label and units in parentheses, e.g., `NDVI (Unitless)` or `Yield (t/ha)`.
* **Legend Placement:** Place legends where they do not obscure data. Using `frameon=True` in the style will add a professional border.

## 4. Final Export Checklist

Before saving, ensure you use `tight_layout` to prevent axis labels from being cut off at the margins.

```python
# Clean up white space
plt.tight_layout()

# Save for LaTeX
plt.savefig('my_figure.pdf', format='pdf', bbox_inches='tight', dpi=300)
```

## 5. Summary Table of Styles

| Style Component | Academic Requirement | Python Implementation |
| --- | --- | --- |
| **Font** | Serif (Times New Roman) | `plt.style.use('science')` |
| **DPI** | Minimum 300 for print | `dpi=300` in savefig |
| **Width** | Matches column width | `figsize=(width, height)` |
| **Lines** | Consistent thickness | Handled by `ieee` style |

## 6. Template Script

A ready-to-use template script is provided in `examples/template.py`.
