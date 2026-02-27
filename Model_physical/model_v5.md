This is the final logic layer for your model. You are moving from a **"Grass Growing Model"** (calculating generic biomass) to a true **"Crop Yield Model"** (calculating grain).

To do this effectively without complex calibration, we will apply a **Phased Stress Approach**. We will treat the plant differently before and after the Silking date you have found.

Here is the complete conceptual breakdown of the logic you need to implement.

---

### 1. The Concept: "Source" vs. "Sink"
You must divide the crop lifecycle into two distinct distinct phases based on your **Silking Date** (the 50% GDD point).

*   **Phase 1: The Vegetative Phase (SOS to Silking)**
    *   *Goal:* The plant is building the "factory" (roots, stems, and leaves).
    *   *Role:* This biomass is not the grain itself. It is the infrastructure needed to support the grain later.
    *   *Stress Sensitivity:* Moderate. If the plant is stressed here, it becomes smaller, but it can recover if rains return.

*   **Phase 2: The Reproductive/Grain-Filling Phase (Silking to EOS)**
    *   *Goal:* The plant is pumping carbon into the "sink" (the cob).
    *   *Role:* Photosynthesis during this period contributes almost directly to grain weight.
    *   *Stress Sensitivity:* **Critical.** Stress here stops the grain from filling. There is no recovery from stress in this phase.

---

### 2. The Accumulation Logic (Weighted NPP)

Instead of a flat sum, you will apply a **Stage-Dependent Weighting** to the daily NPP before summing it.

**For Phase 1 (Vegetative): Apply a Lower Weight**
*   Treat NPP generated during this phase as "Structural Biomass."
*   Only a small fraction of this carbon will eventually be remobilized into the grain.
*   *Logic:* Multiply daily NPP by a factor less than 1.0 (e.g., **0.5**). This acknowledges that a huge stalk does not automatically guarantee a huge cob.

**For Phase 2 (Grain Filling): Apply Full Weight**
*   Treat NPP generated during this phase as "Direct Yield Contribution."
*   *Logic:* Multiply daily NPP by **1.0**. Every gram of carbon fixed now is essentially sugar/starch going into the maize kernel.

**The Result:** This shift ensures that a Late Drought (during grain fill) hurts the final yield score *more* than an Early Drought (during seedling stage), which aligns with agronomic reality.

---

### 3. The "Killer" Logic: The Critical Window Penalty

This is the most important addition for capturing the **yield gaps** and **crashes** in Africa. You must define a specific **"Critical Window"** around the Silking date.

*   **The Window:** Define a period of **20 days** centered on your Silking date (10 days before, 10 days after). This covers pollination and fertilization.
*   **The Check:** Calculate the average Stress Factors (Temperature and Water) *specifically* for this 20-day window.

**The Penalties:**
You apply these as multipliers to the Final Yield at the very end of the calculation.

**A. The Heat Sterility Penalty**
*   If the average Temperature during this 20-day window is too high (e.g., daily means > 28°C or max temps > 35°C), pollen becomes sterile.
*   *Logic:* If the Critical Window Temperature is above a threshold, apply a **0.7** or **0.8** multiplier to the final yield. This simulates the "empty cob" effect where biomass is high but grain is missing.

**B. The Drought Abortion Penalty**
*   If the Water Stress ($W_s$) during this 20-day window drops below a critical level (e.g., 0.5), the plant aborts the kernels to save itself.
*   *Logic:* If the Critical Window Water Stress is severe, apply a heavy penalty (e.g., **0.6**) to the final yield.


---

### 4. Summary of the Final Calculation Flow

1.  **Define Dates:** Use your dynamic calendar to find SOS, Silking, and EOS.
2.  **Loop Phase 1 (SOS to Silking):**
    *   Calculate NPP.
    *   Multiply by **0.5** (Vegetative Weight).
    *   Accumulate to Total.
3.  **Loop Phase 2 (Silking to EOS):**
    *   Calculate NPP.
    *   Multiply by **1.0** (Reproductive Weight).
    *   Accumulate to Total.
4.  **Apply Critical Window Check:**
    *   Look at the 20 days around Silking.
    *   Was it too hot? -> Multiply Total by **0.8**.
    *   Was it too dry? -> Multiply Total by **0.6**.
5.  **Final Conversion:**
    *   Convert this weighted Carbon sum into Grain Yield using the Harvest Index (0.35) and Moisture Content constants.

This approach transforms your model from a "Biomass Calculator" into a "Crop Physiological Model" without requiring any new data inputs. It uses the timing you have already derived to apply the right logic at the right time.