# Best Practices

> [!WARNING]  
> When writing code **create your own branch**, then make a pull request when you're ready to merge to avoid merge conflcts.

# Repo Structure 
Code goes in your home folder (on the workstations, SCC, or your device), checked into git and GitHub. Data, checkpoints, intermediate files, and conda enviroments go on SCC.

Basically anything you check into git (because it's small and handwritten) lives in your home folder, everything else is on SCC.

For the SCC stuff, the main distinctions is:
1. Datasets live in `/projectnb/vkolagrp/datasets/`
2. Project-specific large files go in `/projectnb/vkolagrp/projects/adrd_foundation_model`:save models and checkpoints in `checkpoints`, conda environments in `envs` and large output files or processed data files in `intermediate_files`.

There is already a conda env created by Sahana, you can either activate it by hand using 
```
module load miniconda

conda activate /projectnb/vkolagrp/projects/adrd_foundation_model/envs/fmadrd
```
or run the `activate_env.sh` script, which is just those two lines.

The project structure will evolve as needed, but let's start from something like

```
.
├── README.md       
├── LICENSE
├── notebooks       <-- notebooks, figures, reports
├── references      <-- data dictionaries, dataset docs, any other useful docs
└── src             <-- code, tests, 
```

Use the notebooks folder for exploratory stuff and reports (including figures, presentations, markdown).

Spreadsheet with available datasets: [this Google Sheet](https://docs.google.com/spreadsheets/d/1pXnDFDvU572rrZSNdxGasZBxgMU1gO780_Ll7HFy_x8/edit?usp=sharing)
