from pathlib import Path

from joblib import dump
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    iris = load_iris()
    features_train, features_test, labels_train, labels_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(features_train, labels_train)

    accuracy = accuracy_score(labels_test, model.predict(features_test))
    model_path = Path(__file__).with_name("model.pkl")
    dump(model, model_path)

    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()