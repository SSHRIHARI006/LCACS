import httpx
import asyncio
import time

# 1. Initialize our "Global Model" (Just a list of 5 numbers starting at 0.0)
global_weights = [0.0, 0.0, 0.0, 0.0, 0.0]

async def send_job(client, url, data_size, weights):
    payload = {
        "data_chunk_size": data_size,
        "global_weights": weights
    }
    response = await client.post(url, json=payload, timeout=120.0)
    return response.json()

async def run_epoch(epoch, total_data):
    global global_weights
    print(f"\n=== Starting Epoch {epoch} ===")
    start_time = time.time()
    
    # LCACS logic: Assign 80% to Worker 1, 20% to Worker 2
    w1_data = int(total_data * 0.8)
    w2_data = int(total_data * 0.2)

    assignments = [
        {"url": "http://localhost:8001/train", "data": w1_data},
        {"url": "http://localhost:8002/train", "data": w2_data}
    ]

    print(f"Dispatching {w1_data} samples to Worker 1, and {w2_data} to Worker 2...")
    print(f"Current Global Weights: {[round(w, 4) for w in global_weights]}")
    
    async with httpx.AsyncClient() as client:
        tasks = [send_job(client, job["url"], job["data"], global_weights) for job in assignments]
        results = await asyncio.gather(*tasks)

    # --- Federated Averaging (FedAvg) using pure Python ---
    print("\nAggregating weights (FedAvg)...")
    total_processed = sum(res["data_processed"] for res in results)
    
    # Create an empty list to hold the new averages
    new_global_weights = [0.0] * len(global_weights)
    
    for res in results:
        weight_fraction = res["data_processed"] / total_processed
        local_weights = res["local_weights"]
        
        # Multiply each worker's weight by its fraction and add to the total
        for i in range(len(local_weights)):
            new_global_weights[i] += local_weights[i] * weight_fraction

    # Update the global model
    global_weights = new_global_weights
    
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} Complete! Makespan: {epoch_time:.2f} seconds.")

async def main():
    epochs = 3
    total_dataset_size = 1000
    for i in range(1, epochs + 1):
        await run_epoch(i, total_dataset_size)

if __name__ == "__main__":
    asyncio.run(main())