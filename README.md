# Lithuanian Innovation Agency Risk Model

A data science project for risk modeling using Python and R.

## Project Structure

```
├── data/
│   ├── raw/               # Raw input data (Excel files)
│   ├── processed/         # Cleaned and processed data
│   └── interim/           # Intermediate data files
│
├── notebooks/
│   ├── exploration/       # Exploratory Data Analysis (EDA)
│   ├── preprocessing/     # Data cleaning and preparation
│   ├── modeling/          # Model development notebooks
│   └── visualization/     # Data visualization notebooks
│
├── src/
│   ├── python/            # Python source code
│   └── r/                 # R source code
│
├── docs/                  # Project documentation
│
├── requirements/
│   ├── python.txt         # Python requirements
│   └── r.txt             # R requirements
│
├── config/                # Configuration files
│
└── README.md              # Project overview
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

# Pycharm/DataSpell Specific Setup

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

