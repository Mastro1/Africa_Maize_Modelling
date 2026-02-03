### The Strategy: "Constrained Search"

You will not search the whole year (which risks finding weeds or the wrong season). Instead, you will use the **Fixed Calendar** to define a **"Search Window,"** and then use the **NDVI** to find the specific **"Green-up"** and **"Brown-down"** dates within that window.

---

### The Algorithm (Step-by-Step for one season)

#### Step 1: Define the "Search Window"
Use your CSV crop calendar (`Maize_1_planting`, `Maize_1_endofseaso`).
*   **Window Start:** Fixed Planting Date **minus 30 days** (Allow for early rains).
*   **Window End:** Fixed Harvest Date **plus 30 days** (Allow for late maturation).
*   *Action:* Slice your dataframe to look only at data inside this buffer zone.

#### Step 2: Smooth the Data
Raw NDVI/FPAR is jagged. You need a smooth curve to find the intersection points cleanly.
*   **Action:** Apply **Savitzky-Golay filtering** (window length $\approx$ 30 days, polyorder 2).

#### Step 3: Determine the "Amplitude" (The Height of the Season)
Within your Search Window, find the min and max values.
1.  **Baseline ($V_{min}$):** The minimum NDVI value in the window (usually at the start).
2.  **Peak ($V_{max}$):** The maximum NDVI value in the window.
3.  **Amplitude ($Amp$):** $V_{max} - V_{min}$.

#### Step 4: Define the Thresholds (The 20% Rule)
We define the season as "active" when the vegetation is at **20%** of its seasonal growth height.
$$Threshold = V_{min} + (0.20 \times Amp)$$

#### Step 5: Find the Intersection (SOS & EOS)
1.  **SOS (Start of Season):** Search **backward** from the Peak ($V_{max}$). The first day where the value drops below the `Threshold`.
2.  **EOS (End of Season):** Search **forward** from the Peak ($V_{max}$). The first day where the value drops below the `Threshold`.

### Step 6: Fail-Safe Detection of SOS and EOS  
Occasionally, the smoothed time series may not cross the 20% threshold on one or both sides of the peak. When this occurs, a conservative fallback procedure ensures that both seasonal boundaries remain well defined.

1. **Segment the Search Window:**  
   Use the position of the peak within the window to divide the series into two parts:  
   * a **pre-peak window** for the SOS search,  
   * a **post-peak window** for the EOS search.

2. **Identify Local Minimum Values:**  
   Within each of these sub-windows, determine the minimum NDVI value. This captures baseline vegetation levels relevant to each side of the season.

3. **Compute a Localized Fail-Safe Threshold:**  
   For each sub-window, define a substitute threshold:  
   \[
   V_{\min}^{\text{sub}} + 0.20 \times V_{\min}^{\text{sub}}
   \]  
   This ensures that the threshold reflects minimal but meaningful vegetation activity.

4. **Assign SOS and EOS:**  
   * **SOS:** The earliest date in the pre-peak window where the smoothed NDVI meets or exceeds the fail-safe threshold.  
   * **EOS:** The latest date in the post-peak window where the smoothed NDVI meets or exceeds the fail-safe threshold.

This mechanism prevents missing values and stabilizes seasonal boundary detection in years with weak or irregular vegetation signals.

---

### Step 7: Visualization and Diagnostic Validation  
Before linking this method to the larger model, create diagnostic figures to verify that the detection procedure functions properly under different annual conditions.

1. **Temporal Coverage:**  
   Generate visualizations for four representative years (e.g., 2005, 2010, 2015, 2020).

2. **Spatial Scope:**  
   Restrict the analysis to a single location with the largest crop-area value. Ensure that only observations with valid `admin_2` entries are included.

3. **Elements to Display:**  
   Each plot should include  
   * the raw NDVI values,  
   * the smoothed NDVI curve,  
   * the Search Window boundaries,  
   * the minimum–maximum range and the associated thresholds,  
   * the detected SOS and EOS (whether primary or fail-safe),  
   * clear annotations marking the detected dates.

This validation step provides visual confirmation that each component of the algorithm performs as expected and helps identify anomalies before integration into the production workflow.


### Why this fixes your model

1.  **Variability:** In a drought year, the plant grows slowly. The "20% threshold" will be reached **later**. The season will be shorter. Your model will sum fewer days of NPP. **Result: Lower Yield (Correct).**
2.  **Precision:** You are no longer summing "soil noise" before the plant emerges. You start counting exactly when the satellite sees the plant doing work.
3.  **Safety:** By using the "Known Calendar" as a window, you ensure you don't accidentally detect a forest greening up or a secondary weed flush. You force the model to look only where the crop *should* be.
4. **Max crop area:** We are going to use the max crop area location to analyze the timeseries. Make sure that you filter the locations to keep only those that have a value in "admin_2" column: df = df[df['admin_2'].notna()]