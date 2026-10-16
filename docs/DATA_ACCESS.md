# Data access

No raw data is stored in this repository.

| Project | Dataset | Size | How to obtain |
|---|---|---|---|
| 1 | PLCO lung cancer screening trial extract (`lung_prsn.csv`, ~77 MB, ~155k participants) | 77 MB | Distributed to enrolled students via the course site; originates from the NCI Cancer Data Access System (CDAS). Request access from NCI CDAS for independent use. Place at `project1-lung-cancer-risk/lung_prsn.csv`. |
| 2 | PathMNIST (MedMNIST v2) | 205 MB | Auto-downloaded by `medmnist` on first run, or from https://medmnist.com/ |
| 3 | INSPIRE v1.3 (perioperative research dataset, Seoul National University Hospital) | multi-GB | Access via PhysioNet under its data-use terms (https://doi.org/10.13026/46m4-f655). Do not redistribute. |

PLCO and INSPIRE contain (de-identified) human-subject data governed by data-use agreements;
they are intentionally excluded from version control via `.gitignore`.
