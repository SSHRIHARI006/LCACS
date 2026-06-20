import os
import time
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

WORKER_ID = os.getenv("WORKER_ID", "unknown")
COMPUTE_POWER = float(os.getenv("COMPUTE_POWER", 1.0))
BANDWIDTH = float(os.getenv("BANDWIDTH", 10.0))

class TrainingPayload(BaseModel):
    start_row: int
    end_row: int
    global_weights: List[float]
    learning_rate: float

@app.get("/health")
def health():
    return {
        "worker_id": WORKER_ID,
        "compute_power": COMPUTE_POWER,
        "bandwidth": BANDWIDTH
    }

@app.post("/train")
def train_chunk(payload: TrainingPayload):
    data_size = payload.end_row - payload.start_row
    print(f"[Worker {WORKER_ID}] Reading rows {payload.start_row} to {payload.end_row}")
    
    time.sleep((data_size * 0.1) / BANDWIDTH)
    
    df = pd.read_csv('/app/data/dataset.csv')
    chunk = df.iloc[payload.start_row:payload.end_row]
    
    w_sqft, w_beds, bias = payload.global_weights
    lr = payload.learning_rate
    
    sqft_norm = chunk['sqft'] / 5000.0
    beds_norm = chunk['bedrooms'] / 10.0
    price_norm = chunk['price'] / 1000000.0
    
    for _, row in chunk.iterrows():
        prediction = (w_sqft * (row['sqft']/5000.0)) + (w_beds * (row['bedrooms']/10.0)) + bias
        error = prediction - (row['price']/1000000.0)
        
        w_sqft -= lr * error * (row['sqft']/5000.0)
        w_beds -= lr * error * (row['bedrooms']/10.0)
        bias -= lr * error
        
    time.sleep((data_size * 0.01) / COMPUTE_POWER)
    
    return {
        "worker_id": WORKER_ID,
        "data_processed": data_size,
        "local_weights": [w_sqft, w_beds, bias]
    }
