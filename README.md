# Clinical Machine Learning Portfolio — CPH 100A

> Three end-to-end projects applying machine learning to real clinical problems: **lung cancer screening risk prediction**, **deep learning for histopathology images**, and **validation + prediction on a large perioperative (surgical) dataset**.

Coursework for **CPH 100A** (UC Berkeley, Fall 2025). Projects 1 and 3 were completed in teams; Project 2's models were implemented individually with a team analysis.

| | Project | Problem | Methods | Headline result |
|---|---|---|---|---|
| 1 | [Lung cancer risk models & screening simulation](project1-lung-cancer-risk/) | Predict lung cancer from PLCO questionnaires; compare against NLST screening criteria | Logistic regression + SGD written from scratch in NumPy, custom vectorizer, parallel grid-search dispatcher | **Test AUC 0.816** (val 0.833); at NLST-matched specificity, sensitivity **0.695 vs 0.667** |
| 2 | [PathMNIST classification & segmentation](project2-pathmnist-deep-learning/) | Classify 9 tissue types; localize synthetic boxes | PyTorch: Linear / MLP / CNN classifiers; Linear / MLP / CNN / U-Net segmentation, Dice+BCE loss | U-Net reaches IoU ≈ 1.0 on most sample images vs ≈ 0.1–0.4 for linear/MLP baselines |
| 3 | [INSPIRE perioperative dataset](project3-inspire-perioperative/) | Audit data integrity; reproduce a published mortality model; build a new ICU-admission model | pandas, XGBoost, time-aware table joins, Charlson index, leakage analysis | Reproduced mortality AUC 0.96 (published 0.944); **ICU-admission AUC 0.869** using only pre-operative data |

## Skills demonstrated

- **ML fundamentals from scratch:** gradient derivation, mini-batch SGD with L2 regularization, feature normalization, one-hot/ordinal encoding, missing-data handling
- **Experiment management:** multiprocessing grid-search dispatcher with per-run logs and CSV aggregation
- **Deep learning (PyTorch):** CNNs with batch norm, encoder–decoder and U-Net with skip connections, Dice + BCE loss, IoU evaluation
- **Clinical evaluation:** ROC / precision–recall analysis, sensitivity/specificity/PPV, operating-point matching against a clinical guideline (NLST), subgroup fairness analysis
- **Data auditing & leakage detection:** schema checks, timestamp consistency, physiologic range validation, identifying intra-operative leakage in a published model
- **Scientific communication:** written reports with figures, limitations, and contribution statements

## Repository structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── docs/
│   └── DATA_ACCESS.md                    # how to obtain each dataset (none are committed)
├── project1-lung-cancer-risk/
│   ├── logistic_regression.py            # SGD logistic regression (NumPy)
│   ├── vectorizer.py                     # numerical / categorical / ordinal featurization
│   ├── dispatcher.py                     # parallel hyperparameter grid search
│   ├── main.py                           # train + evaluate
│   ├── grid_search.json, requirements.txt, check_installation.py
│   ├── results/                          # grid_results.csv, results.json
│   ├── figures/                          # age-only L2 vs. AUC plot
│   └── report/                           # team write-up (PDF)
├── project2-pathmnist-deep-learning/
│   ├── classification_{models,train,main,dataset}.py
│   ├── segmentation_{models,train,main}.py
│   ├── synthetic_data.py, metrics.py, requirements.txt
│   └── visualizations/                   # predictions for each segmentation model
└── project3-inspire-perioperative/
    ├── notebooks/                        # EDA, reference-model recreation, our ICU model
    └── report/                           # final report (PDF)
```

## Quick start

Each project has its own `requirements.txt` and README with exact commands. Datasets are **not** included — see [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md).

```bash
git clone https://github.com/<your-username>/cph100-clinical-ml-portfolio.git
cd cph100-clinical-ml-portfolio/project1-lung-cancer-risk   # or project2-... / project3-...
conda create -n cph100 python=3.10 -y && conda activate cph100
pip install -r requirements.txt
```

## Key findings at a glance

**Project 1 — Lung cancer risk.** A logistic regression with age, pack-years, sex, race, smoking status, family history, marital status, education and age level reaches AUC 0.816 on a held-out test set (age alone: ≈0.60). Matched to NLST's specificity (0.803), the model detects 69.5% of cancers versus NLST's 66.7%. Subgroup AUCs differ by more than 0.05 across education level, smoking status and NLST eligibility, which the report links to small subgroup sizes and the very low cancer prevalence (PPV ≈ 7.5% at the NLST operating point).

**Project 2 — Medical imaging.** A 4-block batch-normalized CNN classifier and a skip-connection U-Net, compared against linear and MLP baselines. On the sample images, the skip-connection U-Net localized boxes most accurately (see the qualitative comparison in the Project 2 README).

**Project 3 — INSPIRE.** Data audit found undocumented time units, inverted timestamp pairs (<1%), and a large gap between all-cause and in-hospital death times. The published mortality model includes anesthesia duration, which is only known after surgery; removing that feature from our ICU-admission model alone dropped AUC to 0.79; adding pre-operative ward vitals, the Charlson Comorbidity Index and immunosuppressant/anticoagulant flags recovered performance to 0.869 without using any post-operative information.

## My contributions and attribution

- **Team work:** Project 1 write-up authored with Jake Pastoria, Asli Alpman, Majed AlFaraj and Viraj Gupta. Project 3 completed with Majed AlFaraj, Jake Pastoria and Viraj Gupta.
- **Course scaffolding:** Starter code and project specifications come from the CPH 100A course release ([yala/CPH100_25_release](https://github.com/yala/CPH100_25_release)). Files provided by the instructors (e.g., `synthetic_data.py`, `classification_dataset.py`, `metrics.py`, `check_installation.py`, the overall script structure) are included for completeness; student-implemented parts are the model, vectorizer, dispatcher, training-loop and analysis code.
- **AI-tool use:** The course permitted AI assistance for the vectorizer, dispatcher, utilities and team analyses, and discouraged it for core model/gradient implementations.

## Limitations and next steps

- Project 1's train/val/test split is a single random split (seed 0); no cross-validation or confidence intervals on subgroup AUCs.
- Project 3's XGBoost experiments used a single 70/30 split with the same held-out set for early stopping and reporting, so reported AUCs may be slightly optimistic; a three-way split or cross-validation is the first fix.
- Project 2's repository stores qualitative prediction figures; run logs with aggregate accuracy/IoU should be added alongside them.

## References

1. Lim L. et al. *INSPIRE, a publicly available research dataset for perioperative medicine.* Sci Data 11 (2024). https://doi.org/10.1038/s41597-024-03517-4
2. Yang J. et al. *MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Sci Data 10 (2023).
3. National Lung Screening Trial Research Team. *Reduced lung-cancer mortality with low-dose computed tomographic screening.* N Engl J Med 365 (2011).
4. Ronneberger O. et al. *U-Net: Convolutional networks for biomedical image segmentation.* MICCAI 2015.

## License

MIT for original code (see [LICENSE](LICENSE)); datasets and course-provided scaffolding remain under their original terms.
