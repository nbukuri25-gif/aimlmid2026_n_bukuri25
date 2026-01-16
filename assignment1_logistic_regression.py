# =========================
# Assignment 1 - Multiclass Logistic Regression (Colab)
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -------------------------
# 1) Load CSV
# -------------------------
# Update this filename to match what you uploaded in Colab
CSV_PATH = "points_raw.csv"

df = pd.read_csv(CSV_PATH)

# Expected columns:
# - x
# - y
# - class  (may contain empty/NaN for excluded points)
required_cols = {"x", "y", "class"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

# -------------------------
# 2) Clean / filter data
# -------------------------
# Keep only rows with a valid class label (1,2,3)
df["class"] = pd.to_numeric(df["class"], errors="coerce")  # empty -> NaN
df_train = df.dropna(subset=["class"]).copy()
df_train["class"] = df_train["class"].astype(int)

# Optional: enforce allowed classes
df_train = df_train[df_train["class"].isin([1, 2, 3])].copy()

print("Total rows in CSV:", len(df))
print("Rows used for training:", len(df_train))
print("Class counts:\n", df_train["class"].value_counts().sort_index())

X = df_train[["x", "y"]].values
y = df_train["class"].values

# -------------------------
# 3) Train multinomial logistic regression
# -------------------------
# Scaling is recommended because x and y ranges differ
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        random_state=42
    ))
])

model.fit(X, y)

lr = model.named_steps["lr"]
scaler = model.named_steps["scaler"]

print("\n--- Model coefficients (on standardized features) ---")
print("Classes:", lr.classes_)
print("Intercepts:", lr.intercept_)
print("Coefficients [per class -> (x, y)]:\n", lr.coef_)

# If you want coefficients in ORIGINAL feature space (approximate conversion):
# For multinomial LR: decision is based on (w_scaled * ((x - mean)/std)) + b_scaled
# Convert:
# w_orig = w_scaled / std
# b_orig = b_scaled - sum(w_scaled * mean / std)
means = scaler.mean_
stds = scaler.scale_

w_scaled = lr.coef_
b_scaled = lr.intercept_

w_orig = w_scaled / stds
b_orig = b_scaled - np.sum((w_scaled * means) / stds, axis=1)

print("\n--- Coefficients converted to original x,y scale ---")
for cls, w, b in zip(lr.classes_, w_orig, b_orig):
    print(f"Class {cls}: intercept={b:.6f}, w_x={w[0]:.6f}, w_y={w[1]:.6f}")

# -------------------------
# 4) Visualization: points + decision regions
# -------------------------
# Create a grid for decision regions
x_min, x_max = df_train["x"].min() - 1, df_train["x"].max() + 1
y_min, y_max = df_train["y"].min() - 1, df_train["y"].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)
grid = np.c_[xx.ravel(), yy.ravel()]
pred = model.predict(grid).reshape(xx.shape)

plt.figure(figsize=(10, 7))

# Decision regions (no manual colors specified; matplotlib picks defaults)
plt.contourf(xx, yy, pred, alpha=0.25)

# Scatter points by class
for cls in sorted(df_train["class"].unique()):
    subset = df_train[df_train["class"] == cls]
    plt.scatter(subset["x"], subset["y"], label=f"Class {cls}", s=60)

plt.title("Assignment 1: Multinomial Logistic Regression (Decision Regions)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# 5) (Optional) Overlay the given line from the assignment page
# -------------------------
# If you want, set these to your line points:
LINE_START = (3.62, 0.5)
LINE_END   = (24.48, 7.54)

xs = np.array([LINE_START[0], LINE_END[0]])
ys = np.array([LINE_START[1], LINE_END[1]])

plt.figure(figsize=(10, 7))
plt.scatter(df_train["x"], df_train["y"], s=60)

plt.plot(xs, ys, linewidth=3, label="Given line (start→end)")

plt.title("Assignment 1: Training Points + Given Line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
