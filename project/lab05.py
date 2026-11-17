"""Практична 5: MLP, backpropagation та регуляризація, лише NumPy."""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from common import (
    folder,
    savefig,
    dump,
    image_data,
    numerical_gradient,
    classification,
    cm_plot,
)


class MLP:
    def __init__(
        self,
        widths=(784, 64, 48, 32, 10),
        seed=42,
        l1=0.0,
        l2=0.0,
        dropout=0.0,
        weight_clip=None,
        gradient_clip=None,
        dtype=np.float32,
    ):
        self.widths = list(widths)
        self.seed = seed
        self.l1 = l1
        self.l2 = l2
        self.dropout = dropout
        self.weight_clip = weight_clip
        self.gradient_clip = gradient_clip
        self.dtype = dtype
        rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed + 1000)
        self.weights = [
            (rng.normal(size=(a, b)) * np.sqrt(2 / a)).astype(dtype)
            for a, b in zip(widths[:-1], widths[1:])
        ]
        self.biases = [np.zeros(b, dtype=dtype) for b in widths[1:]]
        self.clip_steps = 0
        self.steps = 0

    def forward(self, x, training=False):
        self.activations = [x]
        self.z_values = []
        self.masks = []
        a = x
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = a @ w + b
            a = np.maximum(z, 0)
            mask = None
            if training and self.dropout:
                mask = (self.rng.random(a.shape) >= self.dropout).astype(self.dtype) / (
                    1 - self.dropout
                )
                a = a * mask
            self.z_values.append(z)
            self.activations.append(a)
            self.masks.append(mask)
        z = a @ self.weights[-1] + self.biases[-1]
        z -= z.max(1, keepdims=True)
        exp = np.exp(z)
        p = exp / exp.sum(1, keepdims=True)
        self.z_values.append(z)
        self.activations.append(p)
        return p

    def loss(self, x, y, regularized=True):
        self.forward(x)
        z = self.z_values[-1]
        loss = np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y])
        if regularized:
            loss += self.l1 * sum(
                np.abs(w).sum() for w in self.weights
            ) + self.l2 / 2 * sum((w * w).sum() for w in self.weights)
        return float(loss)

    def backward(self, y):
        delta = self.activations[-1].copy()
        delta[np.arange(len(y)), y] -= 1
        delta /= len(y)
        gw = [None] * len(self.weights)
        gb = [None] * len(self.biases)
        for layer in reversed(range(len(self.weights))):
            gw[layer] = (
                self.activations[layer].T @ delta
                + self.l1 * np.sign(self.weights[layer])
                + self.l2 * self.weights[layer]
            )
            gb[layer] = delta.sum(0)
            if layer:
                delta = delta @ self.weights[layer].T
                if self.masks[layer - 1] is not None:
                    delta *= self.masks[layer - 1]
                delta *= self.z_values[layer - 1] > 0
        return gw, gb

    def step(self, x, y, lr):
        self.forward(x, training=True)
        gw, gb = self.backward(y)
        norm = np.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in gw + gb))
        factor = 1.0
        if self.gradient_clip is not None:
            factor = min(1, self.gradient_clip / max(norm, 1e-12))
        self.clip_steps += int(factor < 1)
        self.steps += 1
        for j in range(len(self.weights)):
            self.weights[j] -= lr * factor * gw[j]
            self.biases[j] -= lr * factor * gb[j]
            if self.weight_clip is not None:
                np.clip(
                    self.weights[j],
                    -self.weight_clip,
                    self.weight_clip,
                    out=self.weights[j],
                )

    def predict_proba(self, x):
        return np.concatenate(
            [self.forward(x[i : i + 512]).copy() for i in range(0, len(x), 512)]
        )

    def diagnose(self, x):
        maxima = [np.full(b, -np.inf) for b in self.widths[1:-1]]
        for start in range(0, len(x), 512):
            self.forward(x[start : start + 512], training=False)
            for j, z in enumerate(self.z_values[:-1]):
                maxima[j] = np.maximum(maxima[j], z.max(0))
        dead = [int((m <= 0).sum()) for m in maxima]
        sizes = self.widths[1:-1]
        return {
            "dead_per_layer": dead,
            "dead_percent_per_layer": [100 * d / n for d, n in zip(dead, sizes)],
            "dead_percent": 100 * sum(dead) / sum(sizes),
            "sparsity_percent": 100
            * sum(int((abs(w) < 0.01).sum()) for w in self.weights)
            / sum(w.size for w in self.weights),
            "weights_at_boundary": sum(
                int(np.isclose(abs(w), 1, atol=1e-7).sum()) for w in self.weights
            ),
            "gradient_clipped_steps": self.clip_steps,
            "total_steps": self.steps,
        }

    def save(self, path):
        config = {
            k: getattr(self, k)
            for k in [
                "widths",
                "seed",
                "l1",
                "l2",
                "dropout",
                "weight_clip",
                "gradient_clip",
            ]
        }
        config["dtype"] = np.dtype(self.dtype).name
        data = {"config": np.array(json.dumps(config))}
        for j, (w, b) in enumerate(zip(self.weights, self.biases)):
            data[f"w{j}"] = w
            data[f"b{j}"] = b
        np.savez(path, **data)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            config = json.loads(str(data["config"]))
            config["dtype"] = np.dtype(config["dtype"])
            model = cls(**config)
            for j in range(len(model.weights)):
                model.weights[j] = data[f"w{j}"].copy()
                model.biases[j] = data[f"b{j}"].copy()
        return model


def run():
    out = folder(5)
    xall, yall = image_data("train")
    xt, yt = image_data("t10k")
    ids = np.arange(len(yall))
    subset, _ = train_test_split(ids, train_size=15000, stratify=yall, random_state=42)
    tr, va = train_test_split(
        subset, test_size=0.2, stratify=yall[subset], random_state=42
    )
    x, y = xall[tr], yall[tr]
    xv, yv = xall[va], yall[va]
    # Чисельна перевірка на малій мережі: double precision, вимкнений Dropout.
    rng = np.random.default_rng(42)
    small = MLP((4, 5, 4, 3), l2=0.003, dtype=np.float64)
    gx = rng.normal(size=(6, 4))
    gy = np.array([0, 1, 2, 0, 1, 2])
    small.forward(gx)
    gw, gb = small.backward(gy)
    grad = numerical_gradient(
        lambda: small.loss(gx, gy), small.weights + small.biases, gw + gb, count=60
    )
    configs = [
        ("Baseline", {}),
        ("L1 0.0001", {"l1": 0.0001}),
        ("L1 0.001", {"l1": 0.001}),
        ("L2 0.0001", {"l2": 0.0001}),
        ("L2 0.01", {"l2": 0.01}),
        ("Dropout 0.1", {"dropout": 0.1}),
        ("Dropout 0.2", {"dropout": 0.2}),
        ("Dropout 0.3", {"dropout": 0.3}),
        ("Dropout 0.5", {"dropout": 0.5}),
        ("Weight clip 1", {"weight_clip": 1.0}),
        ("Gradient clip 0.1", {"gradient_clip": 0.1}),
        ("Gradient clip 1", {"gradient_clip": 1.0}),
        ("L2 + Dropout 0.2", {"l2": 0.0001, "dropout": 0.2}),
    ]
    initial = MLP()
    initial_weights = [w.copy() for w in initial.weights]
    results = {}
    models = {}
    histories = {}
    for number, (name, config) in enumerate(configs):
        model = MLP(**config)
        assert all(np.array_equal(a, b) for a, b in zip(initial_weights, model.weights))
        history = []
        order_rng = np.random.default_rng(42)
        for epoch in range(30):
            ids = order_rng.permutation(len(y))
            for start in range(0, len(y), 128):
                batch = ids[start : start + 128]
                model.step(x[batch], y[batch], 0.05)
            p = model.predict_proba(x)
            pv = model.predict_proba(xv)
            row = {
                "epoch": epoch + 1,
                "train_loss": float(
                    -np.log(np.maximum(p[np.arange(len(y)), y], 1e-30)).mean()
                ),
                "validation_loss": float(
                    -np.log(np.maximum(pv[np.arange(len(yv)), yv], 1e-30)).mean()
                ),
                "train_accuracy": float((p.argmax(1) == y).mean()),
                "validation_accuracy": float((pv.argmax(1) == yv).mean()),
            }
            history.append(row)
        diagnosis = model.diagnose(x)
        results[name] = {
            "config": config,
            "validation_accuracy": row["validation_accuracy"],
            "validation_loss": row["validation_loss"],
            **diagnosis,
        }
        models[name] = model
        histories[name] = history
        model.save(out / f"model_{number:02d}.npz")
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
        for j, ax in enumerate(axes):
            for w, label in [
                (initial_weights[j], "He init"),
                (models["Baseline"].weights[j], "Baseline"),
                (model.weights[j], name),
            ]:
                ax.hist(w.ravel(), bins=60, density=True, histtype="step", label=label)
            ax.set(title=f"Прихований шар {j + 1}", xlabel="Вага", ylabel="Щільність")
        axes[-1].legend(fontsize=7)
        savefig(out, f"weights_{number:02d}")
        print(
            "MLP",
            name,
            results[name]["validation_accuracy"],
            "DEAD",
            diagnosis["dead_per_layer"],
            flush=True,
        )
        dump(out, {"status": "training", "completed": results})
    baseline = histories["Baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for prefix, label in [("train", "Train"), ("validation", "Validation")]:
        axes[0].plot(
            [r["epoch"] for r in baseline],
            [r[prefix + "_loss"] for r in baseline],
            label=label,
        )
        axes[1].plot(
            [r["epoch"] for r in baseline],
            [r[prefix + "_accuracy"] for r in baseline],
            label=label,
        )
    axes[0].set(xlabel="Епоха", ylabel="Cross-entropy без штрафу")
    axes[1].set(xlabel="Епоха", ylabel="Accuracy")
    axes[0].legend()
    axes[1].legend()
    savefig(out, "learning_curves")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_position([0.09, 0.12, 0.52, 0.80])
    for i, (name, r) in enumerate(
        sorted(
            results.items(),
            key=lambda item: item[1]["validation_accuracy"],
            reverse=True,
        )
    ):
        ax.scatter(r["dead_percent"], r["validation_accuracy"])
        ax.annotate(
            name,
            (r["dead_percent"], r["validation_accuracy"]),
            xytext=(1.08, 0.97 - i * 0.075),
            textcoords="axes fraction",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#88939b", "lw": 0.6},
        )
    ax.set(xlabel="Мертві нейрони, % прихованих нейронів", ylabel="Validation accuracy")
    savefig(out, "dead_vs_accuracy")
    # Оцінка Test лише для моделі, вибраної за Validation.
    winner = min(results, key=lambda n: results[n]["validation_loss"])
    best = models[winner]
    p = best.predict_proba(xt)
    test_metrics = classification(yt, p)
    index = [name for name, _ in configs].index(winner)
    restored = MLP.load(out / f"model_{index:02d}.npz")
    restored_p = restored.predict_proba(xt[:256])
    original_p = best.predict_proba(xt[:256])
    error = float(abs(restored_p - original_p).max())
    assert error < 1e-7 and np.array_equal(restored_p.argmax(1), original_p.argmax(1))
    cm_plot(
        out,
        "test_confusion",
        test_metrics["confusion_matrix"],
        [str(i) for i in range(10)],
    )
    np.savez(out / "split_indices.npz", train=tr, validation=va)
    np.savez(out / "test_predictions.npz", probabilities=p, y=yt)
    dump(
        out,
        {
            "status": "complete",
            "source": "Fashion-MNIST, офіційні файли Zalando Research",
            "seed": 42,
            "split": {"train": len(y), "validation": len(yv), "test": len(yt)},
            "architecture": initial.widths,
            "epochs": 30,
            "batch": 128,
            "learning_rate": 0.05,
            "gradient_check": grad,
            "restore_max_error": error,
            "winner": winner,
            "test": test_metrics,
            "experiments": results,
            "history": histories,
        },
    )
    print("LAB 5 COMPLETE", winner, test_metrics["accuracy"], flush=True)


if __name__ == "__main__":
    run()
