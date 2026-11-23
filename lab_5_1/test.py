import time

import httpx


def main() -> None:
    sample = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=10.0) as client:
        response = client.post("/predict", json=sample)
        response.raise_for_status()
        task_id = response.json()["task_id"]
        print(f"Task ID: {task_id}")

        while True:
            time.sleep(2)
            response = client.get(f"/result/{task_id}")
            response.raise_for_status()
            task = response.json()
            status = task["status"]

            if status == "completed":
                print(f"Prediction: {task['result']}")
                break
            if status == "failed":
                raise RuntimeError(f"Prediction failed: {task['error']}")
            if status not in ("queued", "processing"):
                raise RuntimeError(f"Unexpected task status: {status}")
            print(f"Status: {status}")


if __name__ == "__main__":
    main()