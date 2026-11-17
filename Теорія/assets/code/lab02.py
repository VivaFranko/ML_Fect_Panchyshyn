"""Практична 2: NumPy softmax і симуляція FedAvg без передачі сирих даних серверу."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import f_classif
from common import (
    DATA,
    folder,
    savefig,
    dump,
    classification,
    cm_plot,
    numerical_gradient,
)


class SoftmaxRegression:
    def __init__(self, d, c, ridge=0):
        self.w = np.zeros((d, c))
        self.b = np.zeros(c)
        self.ridge = ridge

    def predict_proba(self, x):
        z = x @ self.w + self.b
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)

    def predict(self, x):
        return self.predict_proba(x).argmax(1)

    def loss_grad(self, x, y):
        z = x @ self.w + self.b
        z -= z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        loss = np.mean(lse - z[np.arange(len(y)), y]) + 0.5 * self.ridge * np.sum(
            self.w**2
        )
        p = self.predict_proba(x)
        p[np.arange(len(y)), y] -= 1
        p /= len(y)
        return loss, x.T @ p + self.ridge * self.w, p.sum(0)

    def fit_epoch(self, x, y, lr, batch, rng):
        ids = rng.permutation(len(y))
        for start in range(0, len(y), batch):
            idx = ids[start : start + batch]
            _, gw, gb = self.loss_grad(x[idx], y[idx])
            self.w -= lr * gw
            self.b -= lr * gb

    def state(self):
        return self.w.copy(), self.b.copy()

    def load(self, state):
        self.w, self.b = (v.copy() for v in state)


class Client:
    def __init__(self, x, y, seed):
        self.x = x.copy()
        self.y = y.copy()
        self.rng = np.random.default_rng(seed)

    def statistics(self):
        return len(self.x), self.x.sum(0), (self.x**2).sum(0)

    def scale(self, mean, std):
        self.x = (self.x - mean) / std

    def update(self, state, epochs, lr, ridge):
        model = SoftmaxRegression(state[0].shape[0], state[0].shape[1], ridge)
        model.load(state)
        for _ in range(epochs):
            model.fit_epoch(self.x, self.y, lr, 64, self.rng)
        # Транспортний контракт: тільки n, W і b, жодних X або y.
        return len(self.y), model.state()


class Aggregator:
    def __init__(self, d, c, xv, yv, ridge):
        self.model = SoftmaxRegression(d, c, ridge)
        self.xv = xv
        self.yv = yv

    def round(self, clients, epochs, lr, ridge):
        states = [
            client.update(self.model.state(), epochs, lr, ridge) for client in clients
        ]
        total = sum(n for n, _ in states)
        state = tuple(
            sum(n / total * weights[j] for n, weights in states) for j in range(2)
        )
        self.model.load(state)
        return {
            "validation_loss": float(self.model.loss_grad(self.xv, self.yv)[0]),
            "validation_accuracy": float(
                np.mean(self.model.predict(self.xv) == self.yv)
            ),
        }


def partition(y, k, alpha, seed):
    rng = np.random.default_rng(seed)
    if alpha is None:
        groups = [[] for _ in range(k)]
        for c in np.unique(y):
            ids = rng.permutation(np.flatnonzero(y == c))
            for j, p in enumerate(np.array_split(ids, k)):
                groups[j].extend(p)
        groups = [np.array(g, dtype=int) for g in groups]
    else:
        for attempt in range(100):
            groups = [[] for _ in range(k)]
            for c in np.unique(y):
                ids = rng.permutation(np.flatnonzero(y == c))
                proportions = rng.dirichlet(np.full(k, alpha))
                cut = (np.cumsum(proportions)[:-1] * len(ids)).astype(int)
                for j, p in enumerate(np.split(ids, cut)):
                    groups[j].extend(p)
            if min(map(len, groups)) >= 20:
                break
        groups = [np.array(g, dtype=int) for g in groups]
    flat = np.concatenate(groups)
    assert len(flat) == len(y) and len(np.unique(flat)) == len(y)
    return groups


def central(x, y, xv, yv, lr, ridge, epochs=150):
    m = SoftmaxRegression(x.shape[1], 7, ridge)
    rng = np.random.default_rng(42)
    history = []
    best = np.inf
    for epoch in range(epochs):
        m.fit_epoch(x, y, lr, 64, rng)
        vl = float(m.loss_grad(xv, yv)[0])
        history.append(
            [
                epoch + 1,
                float(m.loss_grad(x, y)[0]),
                vl,
                float(np.mean(m.predict(xv) == yv)),
            ]
        )
        if vl < best:
            best = vl
            state = m.state()
    m.load(state)
    return m, history


def run():
    out = folder(2)
    raw = pd.read_excel(DATA / "Dry_Bean_Dataset.xlsx")
    duplicates = int(raw.duplicated().sum())
    df = raw.drop_duplicates().reset_index(drop=True)
    enc = LabelEncoder()
    y = enc.fit_transform(df.Class)
    xraw = df.drop(columns="Class").to_numpy(float)
    ids = np.arange(len(y))
    pool, test = train_test_split(ids, test_size=0.2, random_state=42, stratify=y)
    train, val = train_test_split(
        pool, test_size=0.1875, random_state=42, stratify=y[pool]
    )
    x0 = xraw[train]
    yp = y[train]
    scaler = StandardScaler().fit(x0)
    x = scaler.transform(x0)
    xv = scaler.transform(xraw[val])
    xt = scaler.transform(xraw[test])
    yv = y[val]
    scores, pvalues = f_classif(x, yp)
    assoc = dict(zip(df.columns[:-1], [float(s) for s in scores]))
    m = SoftmaxRegression(x.shape[1], 7, 0.001)
    m.w[:] = np.random.default_rng(42).normal(0, 0.01, m.w.shape)
    m.b[:] = 0.01
    _, gw, gb = m.loss_grad(x[:35], yp[:35])
    grad = numerical_gradient(
        lambda: m.loss_grad(x[:35], yp[:35])[0], [m.w, m.b], [gw, gb]
    )
    trials = []
    opts = []
    for ridge in [0.0, 0.001, 0.01]:
        for lr in [0.01, 0.04, 0.1]:
            mm, h = central(x, yp, xv, yv, lr, ridge, 120)
            score = classification(yv, mm.predict_proba(xv))
            trials.append({"lr": lr, "lambda": ridge, **score})
            opts.append((score["log_loss"], lr, ridge, mm, h))
    _, lr, ridge, model, history = min(opts, key=lambda z: z[0])
    metrics = {"central": classification(y[test], model.predict_proba(xt))}
    h = np.asarray(history)
    plt.plot(h[:, 0], h[:, 1], label="Train")
    plt.plot(h[:, 0], h[:, 2], label="Validation")
    plt.xlabel("Епоха")
    plt.ylabel("Cross-entropy + L2")
    plt.legend()
    savefig(out, "central_learning")
    settings = [
        ("IID", None, 1),
        ("Dirichlet 5", 5, 1),
        ("Dirichlet 1", 1, 1),
        ("Dirichlet 0.3", 0.3, 1),
        ("Dirichlet 0.3, E=5", 0.3, 5),
        ("Dirichlet 0.3, E=10", 0.3, 10),
    ]
    logs = {}
    distributions = {}
    local_metrics = {}
    checks = {}
    for name, alpha, epochs in settings:
        groups = partition(yp, 5, alpha, 42)
        clients = [Client(x0[idx], yp[idx], 42 + j) for j, idx in enumerate(groups)]
        statistics = [c.statistics() for c in clients]
        total = sum(z[0] for z in statistics)
        mean = sum(z[1] for z in statistics) / total
        var = sum(z[2] for z in statistics) / total - mean**2
        std = np.sqrt(np.maximum(var, 1e-12))
        assert np.allclose(mean, scaler.mean_) and np.allclose(std, scaler.scale_)
        for client in clients:
            client.scale(mean, std)
        server = Aggregator(x.shape[1], 7, xv, yv, ridge)
        log = []
        best = np.inf
        for round_id in range(1, 61):
            row = server.round(clients, epochs, lr, ridge)
            row["round"] = round_id
            log.append(row)
            if row["validation_loss"] < best:
                best = row["validation_loss"]
                state = server.model.state()
        server.model.load(state)
        metrics[name] = classification(y[test], server.model.predict_proba(xt))
        logs[name] = log
        distributions[name] = [
            [int(np.sum(yp[idx] == c)) for c in range(7)] for idx in groups
        ]
        checks[name] = {
            "samples": total,
            "unique_samples": len(np.unique(np.concatenate(groups))),
            "federated_scaler_matches_central": True,
        }
        if name in ["IID", "Dirichlet 0.3"]:
            local_metrics[name] = []
            for client in clients:
                count, state = client.update(
                    (np.zeros_like(model.w), np.zeros_like(model.b)), 120, lr, ridge
                )
                local = SoftmaxRegression(x.shape[1], 7, ridge)
                local.load(state)
                local_metrics[name].append(
                    {"n": count, **classification(y[test], local.predict_proba(xt))}
                )
        print("FL", name, metrics[name]["macro_f1"], flush=True)
    for name, log in logs.items():
        plt.plot(
            [r["round"] for r in log], [r["validation_loss"] for r in log], label=name
        )
    plt.xlabel("Раунд")
    plt.ylabel("Validation cross-entropy + L2")
    plt.legend(fontsize=8)
    savefig(out, "federated_learning")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8))
    for ax, name in zip(axes, ["IID", "Dirichlet 5", "Dirichlet 1", "Dirichlet 0.3"]):
        im = ax.imshow(distributions[name], cmap="Blues", aspect="auto")
        ax.grid(False)
        ax.set(
            title=name, xlabel="Клас", ylabel="Клієнт", xticks=range(7), yticks=range(5)
        )
    savefig(out, "client_class_distribution")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(
        [0.3, 1, 5],
        [
            metrics[n]["macro_f1"]
            for n in ["Dirichlet 0.3", "Dirichlet 1", "Dirichlet 5"]
        ],
        marker="o",
    )
    axes[0].set(xlabel="α Dirichlet", ylabel="Test macro-F1")
    axes[1].plot(
        [1, 5, 10],
        [
            metrics[n]["macro_f1"]
            for n in ["Dirichlet 0.3", "Dirichlet 0.3, E=5", "Dirichlet 0.3, E=10"]
        ],
        marker="o",
    )
    axes[1].set(xlabel="Локальні епохи E, α = 0.3", ylabel="Test macro-F1")
    savefig(out, "federated_effects")
    cm_plot(
        out, "confusion_central", metrics["central"]["confusion_matrix"], enc.classes_
    )
    counts = df.Class.value_counts()
    counts.plot.barh(color="#28649b")
    plt.xlabel("Кількість зерен")
    savefig(out, "class_balance")
    corr = df.drop(columns="Class").corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.grid(False)
    ax.set(
        xticks=range(16),
        yticks=range(16),
        xticklabels=corr.columns,
        yticklabels=corr.columns,
    )
    plt.setp(ax.get_xticklabels(), rotation=65, ha="right")
    fig.colorbar(im, ax=ax)
    savefig(out, "correlations")
    data = {
        "raw_shape": list(raw.shape),
        "duplicates_removed": duplicates,
        "shape": list(df.shape),
        "missing": int(df.isna().sum().sum()),
        "classes": enc.classes_.tolist(),
        "class_counts": counts.to_dict(),
        "split": {"train": len(train), "validation": len(val), "test": len(test)},
        "feature_anova": assoc,
        "features": df.columns[:-1].tolist(),
        "selected_lr": lr,
        "selected_ridge": ridge,
        "trials": trials,
        "gradient_check": grad,
        "global_metrics": metrics,
        "local_metrics": local_metrics,
        "partition_checks": checks,
        "client_distribution": distributions,
        "rounds": 60,
        "clients": 5,
        "history": logs,
    }
    np.savez(out / "split_indices.npz", train=train, validation=val, test=test)
    np.savez(
        out / "central_weights.npz",
        w=model.w,
        b=model.b,
        mean=scaler.mean_,
        scale=scaler.scale_,
        classes=enc.classes_,
    )
    dump(out, data)
    print("LAB 2 COMPLETE", flush=True)
    return data


if __name__ == "__main__":
    run()
