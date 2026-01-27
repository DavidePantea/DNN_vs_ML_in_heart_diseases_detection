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
From the project root (same folder as Dockerfile):
  ```bash
    docker build -t myimage .
  ```

## 3) Run the container (choose model via CLI):
To run a model use the following code:
```bash
    docker run --rm myimage --model <model_name> --epochs <nr_epochs>
```
In case you are unsure what models are available check in src/main.py:
```bash
  if args.model == "a":
        train_a(epochs=args.epochs)
    elif args.model == "b":
        train_b(epochs=args.epochs)
```

## 2) Work on the project
The models should be included in model_a.py and model_by.py
  ```bash
  .
  ├── Dockerfile
  ├── requirements.txt
  └── src
      ├── __init__.py
      ├── main.py
      └── models
          ├── __init__.py
          ├── model_a.py <-- DL Code
          └── model_b.py <-- ML Code

