Here is how we use derivatives to solve your "Fixed Calendar" and "Long Tail" problems.

---

### 1. The Theory: The Biology of Derivatives

We treat your smoothed NDVI curve as a function $f(t)$.

#### **First Derivative ($f'(t)$): The Speed of Growth**
*   **Positive:** The plant is getting greener.
*   **Negative:** The plant is yellowing/dying.
*   **Zero:** The peak of the season (Tasseling/Flowering).

#### **Second Derivative ($f''(t)$): The Acceleration**
*   **Positive:** The growth is speeding up (Explosive early growth).
*   **Negative:** The growth is slowing down (approaching the peak).

---

### 2. Finding SOS (Start of Season)
**Agronomic Goal:** Find **Emergence**. This is when the plant bursts out of the soil. It is not when the plant is just "visible" (20%), but when it starts its exponential growth phase.

**Mathematical Definition:** The **Local Maximum of the 2nd Derivative**.
*   This is the "Knee" of the curve. Before this point, the curve is flat (soil). At this point, the curve bends upward violently.

### 3. Finding EOS (End of Season)
**Agronomic Goal:** Find **Physiological Maturity** (Black Layer). This is when grain filling stops. After this point, the leaves turn yellow rapidly.
*   *Crucially:* We do not want the "Harvest" (when the curve hits the bottom), because the period between Maturity and Harvest contributes **zero** to yield (it's just drying).

**Mathematical Definition:** The **Global Minimum of the 1st Derivative** (in the descent phase).
*   This represents the day of **Maximum Senescence Rate**.
*   The plant is dying faster on this day than any other day.
*   Biologically, this correlates very strongly with the cessation of photosynthesis and the end of the grain-filling period.

---

### 4. Why this solves your "2020 Plot" problem
Look at your 2020 plot (bottom).
*   **The Issue:** The curve stays high (above 20%) for a long time, so your threshold logic failed and you reverted to "Fixed Used."
*   **The Derivative Solution:** Even though the curve is high, it is **sloping downwards**.
    *   Around May/June 2020, the slope ($f'$) becomes steep.
    *   The Derivative method would identify the steep downward slope as the EOS, effectively cutting off the long "tail" of weeds/drying that confused your threshold model.

---

### 5. Implementation Plan (Python)

We will use `numpy.gradient` to calculate the derivatives of your **Smoothed NDVI**.

#### The Algorithm

1.  **Define Window:** Use your Fixed Calendar +/- 30 days (just to isolate the correct season bump), using the CSV crop calendar (`Maize_1_planting`, `Maize_1_endofseaso`).
2.  **Calculate Derivatives:**
    $$1^{st} Deriv = \frac{d(NDVI)}{dt}$$
    $$2^{nd} Deriv = \frac{d(1^{st} Deriv)}{dt}$$
3.  **Find SOS:**
    *   Look at the time *before* the Peak NDVI.
    *   Find the day where **$2^{nd}$ Derivative is Maximum**.
4.  **Find EOS:**
    *   Look at the time *after* the Peak NDVI.
    *   Find the day where **$1^{st}$ Derivative is Minimum** (most negative).

#### Python Code Snippet

You can replace your current detection logic with this:

```python
import numpy as np

def get_phenology_derivatives(df, start_window, end_window):
    """
    Finds SOS and EOS using 1st and 2nd derivatives.
    """
    # 1. Slice data to the broad search window
    # (Avoids picking up previous/next seasons)
    subset = df[(df['doy'] >= start_window) & (df['doy'] <= end_window)].copy()
    
    if subset.empty:
        return np.nan, np.nan

    # 2. Calculate Derivatives
    # 'gradient' calculates central difference
    subset['deriv1'] = np.gradient(subset['ndvi_smoothed'])
    subset['deriv2'] = np.gradient(subset['deriv1'])
    
    # 3. Find the Peak (Max NDVI) to split the season in half
    peak_idx = subset['ndvi_smoothed'].idxmax()
    peak_doy = subset.loc[peak_idx, 'doy']
    
    # -----------------------------------------
    # 4. Find SOS (Emergence) - Max Acceleration BEFORE Peak
    # -----------------------------------------
    # Look only at days before the peak
    growth_phase = subset[subset['doy'] < peak_doy]
    
    if not growth_phase.empty:
        # Find index of max 2nd derivative
        sos_idx = growth_phase['deriv2'].idxmax()
        sos_doy = growth_phase.loc[sos_idx, 'doy']
    else:
        sos_doy = start_window # Fallback
        
    # -----------------------------------------
    # 5. Find EOS (Maturity) - Max Decay Rate AFTER Peak
    # -----------------------------------------
    # Look only at days after the peak
    decay_phase = subset[subset['doy'] > peak_doy]
    
    if not decay_phase.empty:
        # Find index of min 1st derivative (steepest downward slope)
        # We look for the minimum because the slope is negative during senescence
        eos_idx = decay_phase['deriv1'].idxmin()
        eos_doy = decay_phase.loc[eos_idx, 'doy']
    else:
        eos_doy = end_window # Fallback

    return sos_doy, eos_doy
```

### 6. Visualizing the Logic

Imagine the curve in your plot:
1.  **SOS:** Look at the left side of the hill. Find the point where the curve starts curving **upwards** most aggressively (convex). That is the "Knee."
2.  **EOS:** Look at the right side of the hill. Find the steepest slide. Do not wait for it to hit the bottom. Stop at the steepest point.

**Visualization rules:**
The visualizations must be the same as in v1 (Model_physical\verify_dynamic_calendar.py)

### Implementation notes
* Write comprehensive code with comments and docstrings.
* Add some prints to the terminal to check the progress and see if the results make sense.
* Take one location as example and plot the time series at all the steps (Take the following: KEN.22.5_1)
* This project must written only in /Model_physical folder. Keep results in a subfolder called /Results.
* If you need help you ask. If you don't understand something you ask. If you need documentation you ask.