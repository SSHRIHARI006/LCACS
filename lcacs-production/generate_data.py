import pandas as pd
import numpy as np
import os

os.makedirs('data', exist_ok=True)

np.random.seed(42)
sqft = np.random.normal(1500, 500, 1000)
bedrooms = np.random.randint(1, 6, 1000)
price = (sqft * 150) + (bedrooms * 20000) + 50000 + np.random.normal(0, 10000, 1000)

df = pd.DataFrame({'sqft': sqft, 'bedrooms': bedrooms, 'price': price})
df.to_csv('data/dataset.csv', index=False)
print("Real dataset generated at data/dataset.csv")
