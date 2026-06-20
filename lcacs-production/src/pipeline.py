import httpx
import asyncio
import time
import yaml
import pickle
import os
import shutil

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

TOTAL_DATA = config['data']['total_samples']
EPOCHS = config['training']['epochs']
LR = config['training']['learning_rate']
MODEL_DIR = config['output']['model_dir']
alpha = config['lcacs_tuning']['alpha']
beta = config['lcacs_tuning']['beta']

os.makedirs(MODEL_DIR, exist_ok=True)

global_weights = [0.0, 0.0, 0.0]

async def send_job(client, url, start_row, end_row, weights):
    payload = {
        "start_row": start_row,
        "end_row": end_row,
        "global_weights": weights,
        "learning_rate": LR
    }
    response = await client.post(url, json=payload, timeout=120.0)
    return response.json()

async def run_pipeline():
    global global_weights
    
    async with httpx.AsyncClient() as client:
        r1 = await client.get("http://localhost:8001/health")
        r2 = await client.get("http://localhost:8002/health")
        w1_metrics = r1.json()
        w2_metrics = r2.json()
        
    cp1 = w1_metrics["compute_power"]
    bw1 = w1_metrics["bandwidth"]
    cp2 = w2_metrics["compute_power"]
    bw2 = w2_metrics["bandwidth"]
    
    cost1 = alpha * (1.0 / cp1) + beta * (1.0 / bw1)
    cost2 = alpha * (1.0 / cp2) + beta * (1.0 / bw2)
    
    split_ratio = round(cost2 / (cost1 + cost2), 1)
    w1_split = int(TOTAL_DATA * split_ratio)
    
    for epoch in range(1, EPOCHS + 1):
        print(f"=== Epoch {epoch}/{EPOCHS} ===")
        
        assignments = [
            {"url": "http://localhost:8001/train", "start": 0, "end": w1_split},
            {"url": "http://localhost:8002/train", "start": w1_split, "end": TOTAL_DATA}
        ]

        async with httpx.AsyncClient() as client:
            tasks = [send_job(client, job["url"], job["start"], job["end"], global_weights) for job in assignments]
            results = await asyncio.gather(*tasks)

        new_weights = [0.0, 0.0, 0.0]
        for res in results:
            fraction = res["data_processed"] / TOTAL_DATA
            local_weights = res["local_weights"]
            for i in range(3):
                new_weights[i] += local_weights[i] * fraction
                
        global_weights = new_weights
        print(f"Aggregated Weights: {[round(w, 4) for w in global_weights]}")
        
        model_path = os.path.join(MODEL_DIR, f"lcacs_model_epoch_{epoch}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(global_weights, f)
        print(f"Model saved to -> {model_path}")

    print("Training Complete! Packaging final artifacts...")
    zip_name = config['output']['export_zip']
    shutil.make_archive(zip_name, 'zip', MODEL_DIR)
    print(f"Success! Download your zipped model package: {zip_name}.zip")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
