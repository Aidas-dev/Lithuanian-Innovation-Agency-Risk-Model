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
