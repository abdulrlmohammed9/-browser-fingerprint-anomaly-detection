import numpy as np
import pandas as pd
from pathlib import Path
from src.features import UA_FAMILIES

def make_data(n=5000, seed=42, anomaly_rate=0.08):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < anomaly_rate  # simulated test labels only
    width = rng.choice([1366, 1440, 1536, 1920, 2560], n)
    height = rng.choice([768, 900, 864, 1080, 1440], n)
    plugins = rng.poisson(5, n)
    entropy = np.clip(rng.normal(0.55, 0.13, n), 0, 1)
    hour = rng.integers(0, 24, n)
    family = rng.choice(UA_FAMILIES, n, p=[.58, .15, .12, .12, .03])
    width[y] = rng.choice([320, 4000, 4096], y.sum())
    plugins[y] = rng.choice([0, 35, 50], y.sum())
    entropy[y] = rng.uniform(.95, 1, y.sum())
    hour[y] = rng.integers(0, 6, y.sum())
    df = pd.DataFrame(dict(screen_width=width, screen_height=height, plugin_count=plugins,
                           canvas_entropy=entropy, hour=hour, ua_family=family,
                           synthetic_anomaly_label=y.astype(int)))
    return df

if __name__ == '__main__':
    out=Path('data/synthetic_fingerprints.csv');out.parent.mkdir(exist_ok=True)
    make_data().to_csv(out,index=False)
    print(f'Wrote {out}')
