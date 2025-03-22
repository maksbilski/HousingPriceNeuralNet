# Housing Price Classification Model

This project implements a machine learning model to classify residential properties into three price categories based on various features. The model helps potential buyers quickly assess whether a property fits within their budget constraints, making the house-hunting process more efficient and data-driven.

## Project Overview

The model classifies residential properties into three categories:
- **Cheap**: Properties that can be purchased with available cash (up to $100,000)
- **Average**: Properties that require additional financing (up to $350,000 total)
- **Expensive**: Properties beyond the maximum budget

The classification is based on various property features and characteristics, providing a quick assessment of whether a property fits within different budget scenarios.

## Data

The project uses two datasets:
- `train_data.csv`: Training data containing property features and actual sale prices
- `test_data.csv`: Test data for model evaluation

The datasets include various property attributes that influence the final price classification.

## Model Output

The model produces predictions in the following format:
- 0: Cheap (within cash budget)
- 1: Average (requires financing)
- 2: Expensive (beyond maximum budget)

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         housing_price_neural_net and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── housing_price_neural_net   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes housing_price_neural_net a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Requirements

- Python 3.x
- Required packages listed in `requirements.txt`
- Sufficient computational resources for model training

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/housing-price-classifier.git
cd housing-price-classifier
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data:
   - Place your training data in `data/raw/train_data.csv`
   - Place your test data in `data/raw/test_data.csv`

2. Run the training script:
```bash
python -m housing_price_neural_net.modeling.train
```

3. Generate predictions:
```bash
python -m housing_price_neural_net.modeling.predict
```

## Output Format

The predictions are saved in `pred.csv` with the following specifications:
- Single column containing predicted classes
- No headers
- One prediction per line
- Values: 0 (cheap), 1 (average), 2 (expensive)

## Model Performance

The model is evaluated using accuracy metrics for each class, taking into account the imbalanced nature of the dataset. The evaluation focuses on:
- Overall accuracy
- Per-class accuracy
- Handling of class imbalance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
