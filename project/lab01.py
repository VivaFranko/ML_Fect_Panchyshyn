"""Практична 1: лінійна регресія на Ames, три власні оптимізатори NumPy."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from common import DATA, folder, savefig, dump, numerical_gradient


class LinearGD:
    def __init__(self, lr=0.01, batch=None, epochs=500, ridge=0.0, seed=42):
        self.lr, self.batch, self.epochs, self.ridge, self.seed = (
            lr,
            batch,
            epochs,
            ridge,
            seed,
        )

    def predict(self, x):
        return x @ self.w + self.b

    def loss_grad(self, x, y):
        err = self.predict(x) - y
        n = len(y)
        return (
            0.5 * np.mean(err**2) + 0.5 * self.ridge * np.dot(self.w, self.w),
            x.T @ err / n + self.ridge * self.w,
            np.mean(err),
        )

    def fit(self, x, y, xv, yv):
        rng = np.random.default_rng(self.seed)
        self.w = np.zeros(x.shape[1])
        self.b = np.zeros(1)
        best = np.inf
        stale = 0
        self.history = []
        for epoch in range(self.epochs):
            ids = np.arange(len(y)) if self.batch is None else rng.permutation(len(y))
            size = self.batch or len(y)
            for start in range(0, len(y), size):
                batch = ids[start : start + size]
                loss, g, gb = self.loss_grad(x[batch], y[batch])
                self.w -= self.lr * g
                self.b -= self.lr * gb
            train_loss = float(self.loss_grad(x, y)[0])
            val_loss = float(0.5 * np.mean((self.predict(xv) - yv) ** 2))
            self.history.append([epoch + 1, train_loss, val_loss])
            if not np.isfinite(train_loss) or train_loss > 1e12:
                break
            if val_loss < best - 1e-7:
                best = val_loss
                weights = self.w.copy(), self.b.copy()
                stale = 0
            else:
                stale += 1
            if stale >= 80:
                break
        if np.isfinite(best):
            self.w, self.b = weights
        return self


def metrics(y, p):
    ssr = np.sum((y - p) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r = 1 - ssr / sst
    assert np.isclose(r, r2_score(y, p))
    return {
        "r2": float(r),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(np.mean(abs(y - p))),
    }


def run():
    out = folder(1)
    df = pd.read_csv(DATA / "ames_housing.txt", sep="\t")
    ids = np.arange(len(df))
    tr, hold = train_test_split(ids, test_size=0.4, random_state=42)
    va, te = train_test_split(hold, test_size=0.5, random_state=42)
    # Відбір ознак та всі статистики перетворень визначаються лише на Train.
    candidates = [
        "Overall Qual",
        "Gr Liv Area",
        "Garage Cars",
        "Garage Area",
        "Total Bsmt SF",
        "1st Flr SF",
        "Year Built",
        "Full Bath",
        "Year Remod/Add",
        "TotRms AbvGrd",
        "Lot Area",
        "Fireplaces",
        "Mas Vnr Area",
        "BsmtFin SF 1",
    ]
    corr = (
        df.iloc[tr][candidates + ["SalePrice"]]
        .corr()["SalePrice"]
        .drop("SalePrice")
        .sort_values(key=abs, ascending=False)
    )
    numeric = list(corr.index[:10])
    categorical = ["Neighborhood", "Bldg Type", "House Style", "Kitchen Qual"]
    prep = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric,
            ),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                categorical,
            ),
        ]
    )
    x = prep.fit_transform(df.iloc[tr])
    xv = prep.transform(df.iloc[va])
    xt = prep.transform(df.iloc[te])
    y = df.SalePrice.to_numpy(float)
    ym = y[tr].mean()
    ys = y[tr].std()
    yt = (y[tr] - ym) / ys
    yv = (y[va] - ym) / ys
    model = LinearGD()
    model.w = np.zeros(x.shape[1])
    model.b = np.zeros(1)
    _, g, gb = model.loss_grad(x[:15], yt[:15])
    grad = numerical_gradient(
        lambda: model.loss_grad(x[:15], yt[:15])[0],
        [model.w, model.b],
        [g, np.array([gb])],
    )
    trials = []
    selected = {}
    results = {}
    histories = {}
    for name, batch, epochs, rates in [
        ("Batch GD", None, 2200, [0.01, 0.03, 0.08]),
        ("SGD", 1, 220, [0.0005, 0.002, 0.008]),
        ("Mini-batch GD", 64, 650, [0.003, 0.01, 0.03]),
    ]:
        options = []
        for lr in rates:
            m = LinearGD(lr, batch, epochs).fit(x, yt, xv, yv)
            score = metrics(y[va], m.predict(xv) * ys + ym)
            trials.append({"method": name, "lr": lr, "epochs": len(m.history), **score})
            options.append((score["rmse"], m))
        m = min(options, key=lambda z: z[0])[1]
        selected[name] = m
        results[name] = {
            "learning_rate": m.lr,
            "batch_size": batch or len(tr),
            "epochs": len(m.history),
            "train": metrics(y[tr], m.predict(x) * ys + ym),
            "validation": metrics(y[va], m.predict(xv) * ys + ym),
            "test": metrics(y[te], m.predict(xt) * ys + ym),
        }
        histories[name] = m.history
        h = np.asarray(m.history)
        plt.plot(h[:, 0], h[:, 1], label=name)
    plt.xlabel("Епоха")
    plt.ylabel("J = MSE / 2, стандартизована ціна")
    plt.legend()
    savefig(out, "optimization")
    winner = min(selected, key=lambda n: results[n]["validation"]["rmse"])
    m = selected[winner]
    pred = m.predict(x) * ys + ym
    res = y[tr] - pred
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].scatter(pred, res, s=8, alpha=0.4)
    axes[0, 0].axhline(0, color="black", lw=1)
    axes[0, 0].set(xlabel="Прогнозована ціна, USD", ylabel="Залишок, USD")
    stats.probplot(res, plot=axes[0, 1])
    axes[0, 1].set(
        title="Q-Q залишків",
        xlabel="Теоретичний квантиль",
        ylabel="Упорядкований залишок",
    )
    stdres = (res - res.mean()) / res.std()
    axes[1, 0].scatter(pred, np.sqrt(abs(stdres)), s=8, alpha=0.4)
    axes[1, 0].set(
        xlabel="Прогнозована ціна, USD", ylabel="√|стандартизований залишок|"
    )
    axes[1, 1].hist(res, bins=35, color="#28649b")
    axes[1, 1].set(xlabel="Залишок, USD", ylabel="Кількість")
    savefig(out, "diagnostics")
    c = df.iloc[tr][numeric + ["SalePrice"]].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(c, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.grid(False)
    ax.set(
        xticks=range(len(c)),
        yticks=range(len(c)),
        xticklabels=c.columns,
        yticklabels=c.columns,
    )
    plt.setp(ax.get_xticklabels(), rotation=65, ha="right")
    fig.colorbar(im, ax=ax)
    savefig(out, "correlations")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(y[tr], bins=35)
    axes[0].set(xlabel="SalePrice, USD", ylabel="Кількість")
    axes[1].scatter(df.iloc[tr]["Gr Liv Area"], y[tr], s=8, alpha=0.4)
    axes[1].set(xlabel="Житлова площа, ft²", ylabel="SalePrice, USD")
    savefig(out, "eda")
    shapiro = stats.shapiro(res)
    # VIF і Breusch-Pagan є діагностичними допоміжними регресіями, не способом навчання основної моделі.
    xn = prep.named_transformers_["num"].transform(df.iloc[tr][numeric])
    vifs = {}
    for j, name in enumerate(numeric):
        other = np.column_stack([np.ones(len(xn)), np.delete(xn, j, axis=1)])
        coef = np.linalg.lstsq(other, xn[:, j], rcond=None)[0]
        r = metrics(xn[:, j], other @ coef)["r2"]
        vifs[name] = 1 / (1 - r)
    z = np.column_stack([np.ones(len(x)), x])
    res2 = res**2
    coef = np.linalg.lstsq(z, res2, rcond=None)[0]
    bp = len(res) * metrics(res2, z @ coef)["r2"]
    bp_df = np.linalg.matrix_rank(z) - 1
    # Додатковий обов'язковий експеримент: NumPy Ridge, λ обрано на Validation.
    ridge = []
    for strength in [0.001, 0.01, 0.1, 1.0]:
        rm = LinearGD(0.03, 64, 650, ridge=strength).fit(x, yt, xv, yv)
        ridge.append((metrics(y[va], rm.predict(xv) * ys + ym)["rmse"], strength, rm))
    _, strength, rm = min(ridge, key=lambda z: z[0])
    fig, ax = plt.subplots()
    order = np.argsort(abs(m.w))[-15:]
    ax.barh(np.arange(15) - 0.18, m.w[order], height=0.35, label=winner)
    ax.barh(np.arange(15) + 0.18, rm.w[order], height=0.35, label="Ridge")
    ax.set_yticks(range(15), prep.get_feature_names_out()[order])
    ax.set_xlabel("Вага для стандартизованої цілі")
    ax.legend()
    savefig(out, "ridge_weights")
    data = {
        "source": "Ames Housing, De Cock (2011), 2930 спостережень",
        "shape": list(df.shape),
        "split": {"train": len(tr), "validation": len(va), "test": len(te)},
        "numeric": numeric,
        "categorical": categorical,
        "correlations": corr.to_dict(),
        "missing_train": df.iloc[tr][numeric + categorical].isna().sum().to_dict(),
        "encoded_features": x.shape[1],
        "models": results,
        "trials": trials,
        "winner": winner,
        "gradient_check": grad,
        "shapiro": {"statistic": float(shapiro.statistic), "p": float(shapiro.pvalue)},
        "breusch_pagan": {
            "statistic": bp,
            "df": int(bp_df),
            "p": float(stats.chi2.sf(bp, bp_df)),
        },
        "vif": vifs,
        "ridge": {
            "lambda": strength,
            "norm_unregularized": float(np.linalg.norm(m.w)),
            "norm_regularized": float(np.linalg.norm(rm.w)),
            "validation": metrics(y[va], rm.predict(xv) * ys + ym),
            "test": metrics(y[te], rm.predict(xt) * ys + ym),
        },
    }
    np.savez(out / "split_indices.npz", train=tr, validation=va, test=te)
    pd.DataFrame(
        {"actual": y[te], "prediction": m.predict(xt) * ys + ym, "row": te}
    ).to_csv(out / "test_predictions.csv", index=False)
    import joblib

    joblib.dump(prep, out / "preprocessor.joblib")
    np.savez(out / "model_weights.npz", w=m.w, b=m.b, target_mean=ym, target_std=ys)
    dump(out, data)
    print("LAB 1 COMPLETE", winner, results[winner]["test"], flush=True)
    return data


if __name__ == "__main__":
    run()
