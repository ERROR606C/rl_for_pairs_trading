import numpy as np
import pandas as pd


class PairData:
    def __init__(self, path: str):
        df = pd.read_csv(path, index_col=0)
        self.labels = df.columns.tolist()
        self.timestamps = df.index.to_numpy(dtype=np.int64)
        self.s = df.iloc[:, :2].to_numpy(dtype=np.float64).T  # shape (2, N)

    def __len__(self):
        return len(self.s[0])

    def slice(self, start=0, end=None):
        new = object.__new__(PairData)
        new.labels = self.labels
        new.timestamps = self.timestamps[start:end]
        new.s = self.s[:, start:end]
        return new


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path/to/pair.csv>")
        sys.exit(1)

    data = PairData(sys.argv[1])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data.s[0], label=data.labels[0])
    ax.plot(data.s[1], label=data.labels[1])
    ax.set_xlabel("Time step")
    ax.set_ylabel("Close price")
    ax.legend()
    plt.tight_layout()
    plt.show()
