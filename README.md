# Test 1 – Stroke Prediction (Classification)

This project addresses a supervised classification problem using a public healthcare dataset to predict stroke events based on patient features. The dataset and its description is available on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data).

---

## Project Structure

```
.
├── Dockerfile.dev                # Development Docker container
├── docs                          # MkDocs documentation
│   └── index.md
├── mkdocs.yml                    # MkDocs config file
├── poetry.lock                   # Locked dependencies
├── pyproject.toml                # Project dependencies and config
├── README.md                     # This file
├── stroke_predictor              # Main package
│   ├── __init__.py
│   ├── tests                     # Unit tests
│   ├── utils                     # Utils
├── notebooks                     # Jupyter notebooks (EDA)
│   └── eda.ipynb
└── data                          # Raw and processed data
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
