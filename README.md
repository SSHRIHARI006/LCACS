# LCACS Production Environment

This project implements a distributed machine learning training pipeline for the Load-Balanced Collision-Aware Consensus Scheduling (LCACS) algorithm. It consists of multiple containerized worker nodes and a central orchestrator that dynamically allocates workloads based on worker capability metrics.

## Architecture

- Data Layer: Docker shared volumes mount the dataset directly into containers to avoid HTTP bandwidth bottlenecks.
- Compute Layer: Isolated worker nodes expose a REST API to perform gradient updates on specific segments of the dataset.
- Persistence Layer: Central model weights are saved at each epoch as checkpoint files (.pkl) and archived as a zip package on completion.
- Dynamic Scheduler: The central orchestrator queries /health on all worker nodes to discover their processing power and network bandwidth. It computes allocation splits dynamically using the cost formula to minimize overall makespan.

## Structure

```text
lcacs-production/
├── Dockerfile
├── config.yaml
├── docker-compose.yml
├── generate_data.py
├── requirements.txt
├── data/
└── src/
    ├── pipeline.py
    └── worker.py
```

## Setup and Running

1. Generate Dataset
   Run the dataset generator script locally:
   ```bash
   python generate_data.py
   ```

2. Start the Clusters
   Start the Docker worker nodes:
   ```bash
   docker-compose up --build
   ```

3. Execute the Pipeline
   In another terminal session, trigger the training pipeline orchestrator:
   ```bash
   python src/pipeline.py
   ```
