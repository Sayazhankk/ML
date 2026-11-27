import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Где MLflow хранит данные (создастся файл mlflow.db)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 1. Создаем эксперимент
mlflow.set_experiment("iris-experiment")

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

# Гиперпараметры
n_estimators = 100
random_state = 42

# 2. Запускаем run: все, что внутри with, попадет в MLflow
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Считаем метрики
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Test F1-score: {f1:.3f}")

    # Логируем параметры
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)

    # Логируем метрики
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Сохраняем модель в файл (его использует FastAPI) и логируем как артефакт
    joblib.dump({"model": model, "target_names": list(iris.target_names)}, "model.joblib")
    mlflow.log_artifact("model.joblib")
    print("Saved model.joblib")

    # 3. Model Registry: регистрируем модель (версия присваивается автоматически)
    mlflow.sklearn.log_model(
        model,
        name="model",
        serialization_format="cloudpickle",
        registered_model_name="iris_model",
    )

print("Model registered as 'iris_model'")
