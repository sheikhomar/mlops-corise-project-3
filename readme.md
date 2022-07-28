# Project for the "MLOps: From Models to Production" corese

A code repo for the project in week 3 for the [MLOps: From Models to Production](https://corise.com/course/mlops/) course.


## Getting Started

This projects relies on [pyenv](https://github.com/pyenv/pyenv) and [Poetry](https://python-poetry.org/docs/).

1. Install the required Python version:

   ```bash
   pyenv install
   ```

2. Install dependencies

   ```bash
   poetry install --no-dev
   ```

3. Run server locally

   ```bash
   poetry run uvicorn app.server:app --reload
   ```

4. Build Docker image and run container

   ```bash
   docker build --platform linux/amd64 -t mlops-project-3 .
   docker run -p 8085:8080 mlops-project-3
   # Visit Swagger site: http://localhost:8085/docs
   ```
