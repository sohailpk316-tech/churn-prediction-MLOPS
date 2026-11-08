"""Step 3 - exploratory data analysis. Saves figures to artifacts/eda_*.png."""
import matplotlib
matplotlib.use("Agg")                      # no display needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

sns.set_style("whitegrid")
LABELS = {0: "No", 1: "Yes"}


def _save(fig, name):
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    path = config.ARTIFACTS_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"[INFO] Saved {path.relative_to(config.PROJECT_ROOT)}")


def run_eda():
    print("[INFO] Running EDA...")
    df = pd.read_csv(config.CLEAN_DATA_PATH)
    df["Depression"] = df["target"].map(LABELS)
    print(df["Depression"].value_counts().to_string())

    # 1. Target distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="Depression", hue="Depression", palette="Set2", legend=False, ax=ax)
    ax.set_title("Distribution of Depression (target)")
    ax.set_ylabel("Number of students")
    _save(fig, "eda_target_distribution.png")

    # 2. Age distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["age"].dropna(), bins=10, kde=True, color="steelblue", ax=ax)
    ax.set_title("Age distribution")
    _save(fig, "eda_age_distribution.png")

    # 3. Depression by categorical features
    cols = ["gender", "marital_status", "cgpa", "year_of_study",
            "anxiety", "panic_attack", "sought_treatment"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.ravel(), cols):
        order = sorted(df[col].unique())
        sns.countplot(data=df, x=col, hue="Depression", order=order, palette="Set2", ax=ax)
        ax.set_title(f"Depression by {col}")
        ax.tick_params(axis="x", rotation=30)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    _save(fig, "eda_depression_by_feature.png")

    # 4. Correlation of numeric/binary features with the target
    num_cols = config.NUMERIC_FEATURES + config.BINARY_FEATURES + ["target"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix (numeric / binary features)")
    _save(fig, "eda_correlation.png")

    print("[SUCCESS] EDA completed.")


if __name__ == "__main__":
    run_eda()
