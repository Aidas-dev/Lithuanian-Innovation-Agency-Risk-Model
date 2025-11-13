# Lithuanian Innovation Agency Risk Model

A data science project for risk modeling using Python and R.

## Project Structure

```text
├── data/
│   ├── raw/                   # Raw input data (CSV, Excel, JSON)
│   │   ├── firm-adress-data/  
│   │   ├── firm-balance-statements/
│   │   ├── firm-employment-data/ 
│   │   ├── firm-PnL-statements/
│   │   └── firm-registration-dates/
│   │
│   ├── interim/               # Intermediate processed data
│   │   └── joined-balance-and-PnL/
│   │
│   └── processed/             # Final processed datasets
│
├── notebooks/
│   ├── preprocessing/         # Data cleaning notebooks
│   └── exploration/           # Exploratory analysis notebooks
│
├── src/
│   └── python/                # Python source code (e.g., Functions.py)
│
├── docs/                      # Project documentation
│   └── input-data-documentation/
│
├── requirements/              # Dependency files
│   ├── python.txt             
│   └── r.txt                 
│
├── config/                    # Configuration files
│   └── config.ini
│
└── README.md                  # Project overview
```

## Setup Instructions

### Environment Setup (Recommended)

1. Clone this repository
2. Cleanly install (recommended) Mambaconda to system if not already installed: <br>
[Mamba installation guide](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
3. Create and activate environment using mamba:
   ```bash
   mamba env create -f environment.yml
   mamba activate risk-model-env
   ```
4. Register R kernel for Jupyter:
   ```bash
   Rscript -e "IRkernel::installspec(name = 'risk-model-r', displayname = 'R (Risk Model)')"
   ```
5. Start Jupyter notebook:
   ```bash
   jupyter notebook
   ```

### Conda Setup (Alternative)

1. Clone this repository
2. Create and activate conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate risk-model-env
   ```
3. Register R kernel for Jupyter:
   ```bash
   Rscript -e "IRkernel::installspec(name = 'risk-model-r', displayname = 'R (Risk Model)')"
   ```
4. Start Jupyter notebook:
   ```bash
   jupyter notebook
   ```

### Manual Installation (Alternative)

1. Clone this repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements/python.txt
   ```
3. Install R dependencies:
   ```R
   install.packages(readLines("requirements/r.txt"))
   ```
4. Run notebooks from the notebooks/ directory

## Pycharm/DataSpell Specific Setup

You can use Pycharm or DataSpell to run the notebooks. Here are the steps to set up Pycharm/DataSpell.

### If you used mamba:

Pycharm/DataSpell do not support mamba environments. We use a workaround.

1. Open Pycharm/DataSpell
2. Click on "Open" and select the risk-model directory
3. Click on "File" and then "Settings"
4. Click on "Project Interpreter"
5. Click on the "+" sign and select local Python.
6. Choose folder select.
7. Navigate to miniforge3/envs/risk-model/bin/python3.14 and select it.
8. Accept all window prompts.
9. For R interpreter the process is the same,
10. Navigate to miniforge3/envs/risk-model/bin/R and select it.

### If you used conda:

Conda is supported by Pycharm/DataSpell.

1. Open Pycharm/DataSpell
2. Click on "Open" and select the risk-model directory
3. Click on "File" and then "Settings"
4. Click on "Project Interpreter"
5. Click on the "+" sign and select "Conda Environment"
6. Select the "risk-model-env" environment

---

**Note:** The project uses Git LFS for large data files, mamba/conda download Git-LFS to the risk-model environment. <br>
Before reading the large raw files you will need to pull the files using, or adding a script function in notebook before importing.
In base repository folder run:
   ```bash
   git-lfs pull
   ```

