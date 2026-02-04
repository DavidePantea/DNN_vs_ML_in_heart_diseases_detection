# ML/DL Group Project — Running via Docker (Windows + macOS)

This repo is set up so everyone runs the same environment using Docker (same Python + same dependencies), avoiding “works on my machine” issues.

---

## 0) Prerequisites

Install **Docker Desktop**
  - macOS: Docker Desktop for Mac
  - Windows: Docker Desktop (WSL2 recommended)
Make sure Docker is running:
  ```bash
  docker --version
  ```

## 1) Build the Docker image
From the project root, same folder as Dockerfile:
  ```bash
    docker build -t myimage .
  ```
**If you do not have all the packages installed**

## 2) Run the container (choose model via CLI):
To run a model use the following code:
```bash
    docker run --rm myimage --train --model <model>  --epochs <epochs_nr>
```
In case you are unsure what models are available check in src/main.py:
```bash
  if args.model == "a":
        train_a(epochs=args.epochs)
    elif args.model == "b":
        train_b(epochs=args.epochs)
```
## 3) Handling Datasets
Datasets are downloaded authomatically at runtime, and are stored in ./datasets/versions/6/heart_disease_uci.csv.
When uploading they are not included in download


## 4) Project Structure
The models should be included in model_a.py and model_by.py
  ```bash
 .
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── scripts/
│   └── get_dataset.py
├── datasets/              # downloaded automatically (NOT in git)
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── models
│       ├── __init__.py
│       ├── model_a.py     # DL model
│       └── model_b.py     # ML model
```
Ignored files (.gitignore)
```bash
datasets/
runs/
outputs/
.kagglehub/
__pycache__/
*.pyc
.venv/
```

