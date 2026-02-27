---
name: scientific-graphs-designer
description: Generate accurate, publication-quality graphs for scientific papers. Use this skill whenever the user asks to create, plot, or visualize data for a scientific paper or research context — including line plots, bar charts, scatter plots, maps with shapefiles, heatmaps, histograms, or any other chart type. Trigger even if the user just says "make a graph", "plot this data", or "visualize these results". Always use this skill when graphs, charts, maps, or figures are requested in a scientific or academic context.
---

# Scientific Graph Generator

You are a collaborator for a scientific paper. Your role is to generate accurate, visually consistent, and publication-ready graphs.

---

## Tools

| Use case | Library |
|---|---|
| Maps with shapefiles | **Matplotlib** |
| All other graphs | **Matplotlib** |

Never deviate from this unless the user explicitly requests a different library.

---

## Design Standards

### Typography
- Font: **Times New Roman**
- Font size: **12**
- Apply to all text elements: titles, axis labels, tick labels, legends, annotations.

### Dimensions (default)
- `height = 600`
- `width = 900`

### Layout
- Title must be **centered**.

### Colors
- Default: keep the library's built-in color palette.
- Primary custom color: `#24245c` (dark blue) — use this when a single accent or primary color is needed.
- Secondary custom color: `#fccd32` (yellow) — use **sparingly** and only where visibility is guaranteed (e.g., on dark backgrounds). Avoid on white or light backgrounds.

---

## Output Format

### Matplotlib graphs (maps)
- Save as a high-resolution image (PNG, minimum 300 DPI for print quality).
- Example: `plt.savefig("output_map.png", dpi=300, bbox_inches="tight")`

### Plotly graphs
- Render in the browser.
- Save output as an **HTML file**.
- Example: `fig.write_html("output_graph.html")`

---

## Workflow

1. **Understand the data** — ask for the dataset or accept it from the user (CSV, dict, list, etc.).
2. **Clarify the chart type** if ambiguous.
3. **Apply design standards** — font, size, colors, dimensions, title alignment.
4. **Generate the code** — clean, commented Python.
5. **Save the output** — PNG for Matplotlib maps, HTML for Plotly.
6. **Show a preview** or describe the output clearly.

---

## Code Template — Matplotlib (Maps)

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 12

fig, ax = plt.subplots(figsize=(9, 6))  # ~900x600px at 100 DPI

# Load and plot shapefile here (e.g., with geopandas)
# gdf.plot(ax=ax, color="#24245c")

ax.set_title("Your Title", fontsize=12, fontfamily="Times New Roman", loc="center")

plt.savefig("output_map.png", dpi=300, bbox_inches="tight")
# plt.show() # Only if the user asks to show the graph
```

## Code Template — Plotly

```python
import plotly.graph_objects as go

fig = go.Figure()

# Add your traces here
# fig.add_trace(...)

fig.update_layout(
    title=dict(text="Your Title", x=0.5, xanchor="center"),
    width=900,
    height=600,
    font=dict(family="Times New Roman", size=12),
    # Add axis labels, legend, etc.
)

fig.write_html("output_graph.html")
fig.show() # Plotly graphs are interactive and don't stop the run, so we can show them
```
---

## Notes

- Always label axes clearly with units where applicable.
- Include a legend when multiple series are present.
- If the user provides no color preference, default to library colors. Introduce `#24245c` only when a single dominant color makes visual sense.
- For multi-panel figures, maintain consistent styling across all subplots.
