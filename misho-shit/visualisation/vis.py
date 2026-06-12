import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

TRAIN_RATIO = 0.80

# ── paths ─────────────────────────────────────────────────────────────
CSV_A = '../dataset/CORN_USD_2005_2020.csv'
CSV_B = '../dataset/WHEAT_USD_2005_2020.csv'
# ─────────────────────────────────────────────────────────────────────

df_a = pd.read_csv(CSV_A).dropna()
df_b = pd.read_csv(CSV_B).dropna()

n = len(df_a)
print(f"length: {n}")

# ── OLS: A = alpha + beta·B ───────────────────────────────────────────
A = df_a['close']
B = df_b['close']

split = int(n * TRAIN_RATIO)
X = np.column_stack([np.ones(split), B.values[:split]])
beta_ols  = np.linalg.lstsq(X, A.values[:split], rcond=None)[0]
alpha_ols = beta_ols[0]
beta      = beta_ols[1]
print(f'OLS  alpha={alpha_ols:.4f}  beta={beta:.4f}')

spread = A - beta * B

# ── Z-score ───────────────────────────────────────────────────────────
mean = spread.mean()
std  = spread.std()
z    = (spread - mean) / std

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle("Z-Score", fontsize=14, fontweight="bold")

y_min = z.min() * 1.1
y_max = z.max() * 1.1

ax.axhspan( 1.0, y_max, color="red",   alpha=0.25, label="Z > +1.0")
ax.axhspan(y_min, -1.0, color="green", alpha=0.25, label="Z < −1.0")

ax.plot(z.index, z.values, color="dimgrey", lw=1, label="Z-score", zorder=2)

ax.axhline( 1.0, color="red",   lw=1, ls="--")
ax.axhline(-1.0, color="green", lw=1, ls="--")
ax.axhline( 0,   color="grey",  lw=0.6, ls=":")

ax.set_ylim(y_min, y_max)
ax.set_ylabel("Z-score")
ax.set_xlabel("Time point")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()