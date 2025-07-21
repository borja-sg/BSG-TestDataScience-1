# Test 1 – Stroke Prediction (Classification)

This project addresses a supervised classification problem using a public healthcare dataset to predict stroke events based on patient features. The dataset and its description is available on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data).

---

## Project Structure

```
.
├── data/ # Raw and processed data files
├── docs/ # Documentation (MkDocs)
├── notebooks/ # EDA and experimentation notebooks
├── stroke_predictor/ # Source code and utility functions
│ ├── configs/ # YAML configurations
│ ├── optuna_optimization/ # Optuna-based model optimization
│ ├── utils/ # Preprocessing, plotting, and I/O helpers
│ └── tests/ # Unit tests
├── templates/ # HTML templates for the web interface
├── pyproject.toml # Poetry config
├── mkdocs.yml # MkDocs config
└── README.md # Project overview
```

---

## Running Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/borja-sg/BSG-TestDataScience-1
   cd BSG-TestDataScience-1
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Run exploratory data analysis notebooks in `notebooks/` to understand the data.

4. Perform hyperparameter optimization:
   ```bash
   poetry run python stroke_predictor/optuna_optimization/runner.py ./stroke_predictor/configs/optuna.yml
   ```

5. Train the model with the best parameters:
   ```bash
   poetry run python stroke_predictor/train_model.py ./stroke_predictor/configs/train.yml
   ```

6. Launch the web inference application:
   ```bash
   uvicorn stroke_predictor.main:app --host 0.0.0.0 --port 8888 --reload
   ```

7. Access the web interface at:
   ```
   http://localhost:8888/web
   ```


---

## Docker (Development)

Build the development image:

```bash
docker build -f Dockerfile.dev -t stroke-dev .
```

Run container mounting current code:

```bash
docker run --rm -it \
  -v "$(pwd)":/bsg-testdatascience-1 \
  -p 8888:8888 \
  --name stroke-dev \
  stroke-dev
```

---

## EDA

The initial exploratory data analysis is in `notebooks/eda.ipynb`. It loads and explores the dataset structure, distributions, and relationships.

To launch Jupyter:

```bash
poetry run jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

Open the notebook and add the kernel on the port opened before:

```bash
http://localhost:8888
```

---

## Documentation

Documentation is powered by [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

To serve the docs locally:

```bash
poetry run mkdocs serve --dev-addr=0.0.0.0:8888
```

To build the static site:

```bash
poetry run mkdocs build
```

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/borja-sg/BSG-TestDataScience-1
   cd BSG-TestDataScience-1
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Run notebooks or scripts from the root folder.

---

## Optimization and Web Inference

### Run Hyperparameter Optimization with Optuna
```bash
nohup poetry run python stroke_predictor/optuna_optimization/runner.py ./stroke_predictor/configs/optuna.yml &
```

### Start the Optuna dashboard
```bash
optuna-dashboard sqlite:///outputs/optuna_stroke_prediction.db --host 0.0.0.0 --port 8888
```

### Train Model Using Best Parameters
```bash
nohup poetry run python stroke_predictor/train_model.py ./stroke_predictor/configs/train.yml &
```

### Launch Web Inference App
```bash
uvicorn stroke_predictor.main:app --host 0.0.0.0 --port 8888 --reload
```

Open your browser at:
```
http://localhost:8888/web
```

---

## Pre-commit Hooks

Includes code formatting and linting:

- `black`: code formatter
- `ruff`: linter and autofixer
- `mypy`: type checker
- `vulture`: unused code detector

Run manually before commits:

```bash
poetry run pre-commit run --all-files
```

---

## Tests

Tests are located under `stroke_predictor/tests/` and are executed with `pytest`.

Run them with:

```bash
poetry run pytest
```

---

## License

[GNU General Public License v3.0](LICENSE)
