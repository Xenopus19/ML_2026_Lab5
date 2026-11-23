import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel, Field
from enum import Enum
from uuid import uuid4
from typing import Any

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length (см)", examples=[5.1])
    sepal_width: float = Field(..., gt=0, description="Sepal width (см)", examples=[3.5])
    petal_length: float = Field(..., gt=0, description="Petal length (см)", examples=[1.4])
    petal_width: float = Field(..., gt=0, description="Petal width (см)", examples=[0.2])

    def to_feature_vector(self) -> list[list[float]]:
    
        return [[
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width,
        ]]

class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PredictRequest(BaseModel):
    item: IrisFeatures

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    input_data: IrisFeatures
    status: TaskStatus = TaskStatus.QUEUED
    result: Any | None = None
    error: str | None = None

    
tasks: dict[str, Task] = {}
task_queue: asyncio.Queue[Task] = asyncio.Queue()


async def queue_worker(model):
    while True:
        task_data = await task_queue.get()
        try:
            task_data.status = TaskStatus.PROCESSING
            prediction = int(model.predict(task_data.input_data.to_feature_vector())[0])
            task_data.result = prediction
            task_data.status = TaskStatus.COMPLETED
        except Exception as e:
            task_data.status = TaskStatus.FAILED
            task_data.error = str(e)
        finally:
            task_queue.task_done()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(__file__).with_name("model.pkl")
    model = load(model_path)
    print(f"Model loaded from {model_path}")

    worker = asyncio.create_task(queue_worker(model))
    yield
    worker.cancel()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def add_to_queue(item: IrisFeatures):
    task = Task(input_data=item)
    tasks[task.id] = task
    await task_queue.put(task)
    return {"status": "queued", "queue_size": task_queue.qsize(), "task_id": task.id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    if task_id not in tasks:
        return {"status": "not found"}
    task = tasks[task_id]
    return {"status": task.status, "result": task.result, "error": task.error}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 