# ML Model Deployment with FastAPI and Docker

An Iris flower classifier (Random Forest) served via FastAPI and packaged with Docker.

## Project structure
- `train.py` – trains the model and saves `model.joblib`
- `main.py` – FastAPI app (`GET /`, `POST /predict`)
- `requirements.txt` – dependencies
- `Dockerfile` – container instructions

## Run locally
```bash
pip install -r requirements.txt
python train.py
uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs for Swagger UI.

## Run with Docker
```bash
docker build -t ml-fastapi .
docker run -p 8000:8000 ml-fastapi
```
Open http://localhost:8000/docs

## Example requests
```bash
curl http://localhost:8000/
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```
Response: `{"prediction":0,"class_name":"setosa","probability":1.0}`
