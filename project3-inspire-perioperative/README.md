# Project 3 (Final) — INSPIRE: Data Integrity, Model Reproduction, and ICU-Admission Prediction

**INSPIRE** is a public perioperative dataset of ~131,000 surgeries (2011–2020, Seoul National University Hospital) spanning operating rooms, wards and ICUs. We asked three questions:

1. Is the dataset internally consistent and clinically plausible?
2. Do predictive signals reflect real pre-operative information, or artifacts / leakage?
3. Can the published 30-day mortality model be reproduced, and what other clinical prediction tasks does the data support?

Team: Majed AlFaraj, Jake Pastoria, Viraj Gupta. Full report: [`report/CPH100_Final_Project_Writeup.pdf`](report/CPH100_Final_Project_Writeup.pdf). Work split: [`report/CPH100_Final_Project_Contribution_Statement.pdf`](report/CPH100_Final_Project_Contribution_Statement.pdf).

## Findings

### 1. Data integrity audit (`notebooks/cph_final_proj_eda.ipynb`)
- **Schema:** several vague or incorrect variable descriptions (e.g., `height` in `operations`); `*_time` columns carry no units — we inferred **minutes** by reverse-engineering the authors' code.
- **Missingness:** core timestamps (admission, OR in/out) complete; operation/anesthesia start–end pairs have small, unequal missingness; death/ICU/bypass times are mostly missing as expected.
- **Temporal consistency:** a majority of records have overlapping (non-strictly-increasing) required timestamps; start-after-end inversions occur in <1% of pairs and can be filtered. In-hospital vs. all-cause death times differ by a mean of 20.1 and median of 11.2 days.
- **Lab plausibility:** all 38 lab measurements checked against reference ranges (0 physiologically impossible values); 37/38 have some out-of-range but plausible values, as expected in a hospital cohort.

### 2. Reproducing the published model (`notebooks/recreation.ipynb`)
- Restricted to each patient's **first operation**, excluded ASA 6 → 97,260 cases, **268 deaths (0.28%)**.
- Joined the nearest preceding lab (≤ 6 months) to each operation (mean lab recency ≈ 12 days).
- XGBoost with the paper's features and hyperparameters: **AUC 0.96** (published 0.944).
- **Leakage concern:** the reference model uses *duration of anesthesia*, which is only known once surgery ends and so is not a valid pre-operative predictor.

### 3. Our model: pre-operative ICU admission (`notebooks/ourmodel.ipynb`)
- Target: ICU admission after surgery (10,785 / 97,260 = **11.1%** positive vs. 0.28% for mortality → far less class imbalance).
- Removed the leaking feature (AUC fell to **0.79**), then added:
  - pre-operative **ward vitals** (temperature, heart rate, blood pressure, respiratory rate, SpO₂)
  - **Charlson Comorbidity Index** from ICD-10 diagnoses (via `comorbidipy`)
  - **immunosuppressant / anticoagulant** exposure flags from prior medications
- **Final AUC ≈ 0.869** (XGBoost, 500 rounds, lr 0.02, depth 6; 70/30 split).

## Reproduce

Notebooks were authored in Google Colab and mount a Drive folder containing the INSPIRE v1.3 zip. To run elsewhere:

1. Obtain INSPIRE v1.3 from PhysioNet (see `../docs/DATA_ACCESS.md`).
2. Replace the Colab cell that mounts Drive/unzips with a local path to the extracted CSVs (`operations.csv`, `labs.csv`, `vitals.csv`, `diagnosis.csv`, `medications.csv`, `ward_vitals.csv`).
3. Install dependencies and run in order: `cph_final_proj_eda.ipynb` → `recreation.ipynb` → `ourmodel.ipynb`.

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost torch
pip install git+https://github.com/vvcb/comorbidipy.git@c676a8067ce343704d75bcf8180f9baa1ac47664
```

## Limitations and next steps
- A single 70/30 split is used for early stopping and for the reported AUC; a separate validation set or cross-validation would give an unbiased estimate.
- The ICU label column is named `died30` in `ourmodel.ipynb` (carried over from the reproduction code); it encodes ICU admission. Renaming it would avoid confusion.
- Only first operations are modeled, so patients are not followed across repeat surgeries.
- Calibration, subgroup performance and decision-curve analysis were not evaluated.

## Data
INSPIRE (Lim et al., *Sci Data* 2024; PhysioNet doi:10.13026/46m4-f655). Not redistributed here.
