# ML Model Deployment with FastAPI, Docker, Streamlit and MLflow

An Iris flower classifier (Random Forest) served via FastAPI, packaged with Docker,
with a Streamlit frontend and MLflow experiment tracking / model registry.

## Project structure
- `train.py` – trains the model, logs everything to MLflow, saves `model.joblib`
- `main.py` – FastAPI app (`GET /`, `POST /predict`)
- `streamlit_app.py` – simple frontend that calls the API
- `requirements.txt` – dependencies
- `Dockerfile` – container instructions

## Run locally
```bash
pip install -r requirements.txt
python train.py
uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs for Swagger UI.

## Frontend (Streamlit)
Start the API first, then in a second terminal:
```bash
streamlit run streamlit_app.py
```
Open http://localhost:8501, move the sliders and press **Predict**.

## MLflow
`python train.py` creates the experiment `iris-experiment` and logs:
- parameters: `n_estimators`, `random_state`
- metrics: `accuracy`, `f1_score`
- artifacts: `model.joblib` and the MLflow model folder `model`

The model is registered in the Model Registry as `iris_model` (each run creates a new version).

View the results in the MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open http://localhost:5000 -> **Experiments** (runs, params, metrics) and **Models** (`iris_model`, version 1).

## Run with Docker
```bash
docker build -t ml-fastapi .
docker run -p 8000:8000 ml-fastapi
```
Open http://localhost:8000/docs. The Streamlit app can still be started locally and will use this API.

## Example requests
```bash
curl http://localhost:8000/
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```
Response: `{"prediction":0,"class_name":"setosa","probability":1.0}`
