import os
import time
import random
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

WORKER_ID = os.getenv("WORKER_ID", "unknown")
COMPUTE_POWER = float(os.getenv("COMPUTE_POWER", 1.0))
BANDWIDTH = float(os.getenv("BANDWIDTH", 10.0))

class TrainingPayload(BaseModel):
    data_chunk_size: int
    global_weights: List[float]  # Just a simple list of numbers now!

@app.post("/train")
def train_chunk(payload: TrainingPayload):
    print(f"[Worker {WORKER_ID}] Received job: {payload.data_chunk_size} data points.")
    
    # --- 1. Simulate Communication Delay (D / B_i) ---
    data_size_mb = payload.data_chunk_size * 0.1
    transfer_time = data_size_mb / BANDWIDTH
    print(f"[Worker {WORKER_ID}] Simulating download: {transfer_time:.2f} seconds...")
    time.sleep(transfer_time)
    
    # --- 2. Load the Global Model ---
    local_weights = payload.global_weights.copy()

    # --- 3. Simulate Training (Modify the weights slightly) ---
    start_compute = time.time()
    
    # Simulate compute differences (slower workers take longer)
    compute_time = (payload.data_chunk_size * 0.01) / COMPUTE_POWER
    time.sleep(compute_time)
    
    # "Train" by adding a small pseudo-random gradient to the weights
    for i in range(len(local_weights)):
        local_weights[i] += random.uniform(-0.1, 0.1)
    
    total_time = time.time() - start_compute + transfer_time
    print(f"[Worker {WORKER_ID}] Finished in {total_time:.2f} seconds.")

    # --- 4. Return the new weights ---
    return {
        "worker_id": WORKER_ID,
        "data_processed": payload.data_chunk_size,
        "total_time": total_time,
        "local_weights": local_weights
    }