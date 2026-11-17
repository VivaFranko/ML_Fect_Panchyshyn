"""Шляхи, дані й збереження результатів практичних робіт цього курсу."""

from pathlib import Path
import os

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "2")
import json, gzip, zlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    log_loss,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COLORS = ["#28649b", "#c26532", "#397e64", "#735a9d", "#6a7580", "#b74358", "#b59b34"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=COLORS),
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "figure.figsize": (8, 4.6),
    }
)


def stable_hash(word):
    """Стабільний хеш для ініціалізації ембеддингів, незалежний від hash seed Python."""
    return zlib.adler32(word.encode("utf-8"))


def folder(n):
    out = ROOT / f"lab_{n:02d}" / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out


def savefig(out, name):
    plt.tight_layout()
    p = out / (name + ".png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    return p.name


def dump(out, data):
    (out / "results.json").write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=lambda x: (
                x.item()
                if isinstance(x, np.generic)
                else x.tolist()
                if isinstance(x, np.ndarray)
                else str(x)
            ),
        ),
        encoding="utf-8",
    )


def classification(y, p):
    pred = p.argmax(1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, p, labels=np.arange(p.shape[1]))),
        "confusion_matrix": confusion_matrix(
            y, pred, labels=np.arange(p.shape[1])
        ).tolist(),
    }


def cm_plot(out, name, matrix, labels):
    fig, ax = plt.subplots(figsize=(7, 5))
    m = np.asarray(matrix)
    im = ax.imshow(m, cmap="Blues")
    ax.grid(False)
    for i in range(len(m)):
        for j in range(len(m)):
            ax.text(
                j,
                i,
                str(m[i, j]),
                ha="center",
                va="center",
                color="white" if m[i, j] > m.max() / 2 else "#182b3a",
            )
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Прогноз",
        ylabel="Справжній клас",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.colorbar(im, ax=ax)
    return savefig(out, name)


def numerical_gradient(loss, params, grads, seed=42, count=30, epsilon=1e-5):
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(count):
        j = int(rng.integers(len(params)))
        p = params[j]
        idx = np.unravel_index(int(rng.integers(p.size)), p.shape)
        old = p[idx]
        p[idx] = old + epsilon
        plus = loss()
        p[idx] = old - epsilon
        minus = loss()
        p[idx] = old
        numeric = (plus - minus) / (2 * epsilon)
        analytic = grads[j][idx]
        errors.append(
            float(abs(numeric - analytic) / max(1e-8, abs(numeric) + abs(analytic)))
        )
    assert max(errors) < 1e-4, errors
    return {
        "parameters_checked": count,
        "max_relative_error": max(errors),
        "epsilon": epsilon,
    }


def image_data(prefix):
    with gzip.open(DATA / (prefix + "-images-idx3-ubyte.gz"), "rb") as f:
        x = (
            np.frombuffer(f.read(), dtype=np.uint8, offset=16)
            .reshape(-1, 784)
            .astype(np.float32)
            / 255
        )
    with gzip.open(DATA / (prefix + "-labels-idx1-ubyte.gz"), "rb") as f:
        y = np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(int)
    return x, y
