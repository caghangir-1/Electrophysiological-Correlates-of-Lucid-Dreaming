This repository contains the custom preprocessing pipeline and microsaccade detection tool developed for the study:

Demirel, Ç., Gott, J., Appel, K., Lüth, K., Fischer, C., Raffaelli, C., Westner, B., Wang, X., Zavecz, Z., Steiger, A., Erlacher, D., LaBerge, S., Mota-Rolim, S., Ribeiro, S., Zeising, M., Adelhöfer, N., & Dresler, M. (2025). 
**Electrophysiological correlates of lucid dreaming: sensor and source level signatures**. Journal of Neuroscience. https://doi.org/10.1523/JNEUROSCI.2237-24.2025

The custom preprocessing pipeline was developed to enable adaptive artifact suppression across a wide range of EEG recordings—including low-density setups (e.g., 6 channels) and data collected from multiple labs. While designed for EEG, the pipeline is also compatible with MEG datasets, particularly those with long temporal durations, where it may even outperform mid-to-high-density EEG in terms of artifact cleaning.

Besides, micro-saccade detection algorithm were also developed to help determination of potential micro-saccades around eye movements mostly occur in REM sleep & lucid dreaming.

This customly built code was publicly released to support transparency, reproducibility, and future research in the study of conscious awareness during sleep.

For any questions, feel free to reach out: **cagatay.demirel.sci@gmail.com**

---

### 📄 A small note

If you find these methods useful in your own research, please consider citing the associated paper: https://doi.org/10.1523/JNEUROSCI.2237-24.2025

---

### ⚠️ Axis labeling & unit scale clarification in the published paper (Figures 4B2 & 6C)

In the published paper, two figures apply percent labels to values expressed as ratios:

* Figure 4B2: The y-axis label for Relative power change [%]
* Figure 6C: dPLI change [%]

In both cases, the plotted values reflect unitless ratio changes **(e.g., –0.5 = –50% change)**, but were labeled with a percent sign (%) based on a semantic interpretation of percent change, rather than numerical scaling. The values were not multiplied by **100** and should be interpreted as ratios rather than literal percentages. 

**The normalization approach used here follows the same logic as MNE’s % (percent) mode in mne.baseline.rescale()—that is, a fractional baseline correction computed as (data − baseline) / baseline, without multiplying by 100: https://mne.tools/stable/generated/mne.baseline.rescale.html**
