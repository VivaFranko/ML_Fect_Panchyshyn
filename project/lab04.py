"""Практична 4: Annealing, чесна вкладена CV, Optuna та пояснення моделей.

Запуск: python project/lab04.py --number N
N обов'язково є підтвердженим номером студента, а не довільним seed.
"""

import argparse, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    cross_val_score,
    cross_val_predict,
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import optuna, shap, cloudpickle
from common import DATA, folder, savefig, dump, classification, cm_plot

NUMERIC = ["carbon", "hardness", "strength", "thick", "width", "len"]


class TrainColumnCleaner(TransformerMixin, BaseEstimator):
    def fit(self, x, y=None):
        self.columns_ = x.columns[x.nunique(dropna=True) > 1].tolist()
        self.removed_ = [c for c in x if c not in self.columns_]
        return self

    def transform(self, x):
        return x[self.columns_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_, dtype=object)


def read_annealing():
    text = (DATA / "anneal.names").read_text(encoding="utf-8")
    attribute_block = text.split("7. Attribute Information:", 1)[1].split(
        "classes:", 1
    )[0]
    names = re.findall(r"^\s*\d+\. ([^:\n]+):", attribute_block, re.M)
    assert len(names) == 38, names
    frame = pd.concat(
        [
            pd.read_csv(
                DATA / name,
                header=None,
                names=names + ["class"],
                na_values="?",
                keep_default_na=False,
            )
            for name in ["anneal.data", "anneal.test"]
        ],
        ignore_index=True,
    )
    for col in names:
        if col in NUMERIC:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
        else:
            frame[col] = frame[col].astype(object).where(frame[col].notna(), np.nan)
    assert len(frame) == 898 and len(names) == 38
    return frame


def pipeline(name, seed, indicators=True, params=None):
    params = params or {}
    if name == "DT":
        model = DecisionTreeClassifier(random_state=seed, **params)
    elif name == "RF":
        model = RandomForestClassifier(random_state=seed, n_jobs=2, **params)
    else:
        model = XGBClassifier(
            random_state=seed, n_jobs=2, eval_metric="mlogloss", **params
        )
    transforms = [
        (
            "num",
            SimpleImputer(strategy="median", keep_empty_features=True),
            make_column_selector(dtype_include="number"),
        ),
        (
            "cat",
            Pipeline(
                [
                    (
                        "impute",
                        SimpleImputer(
                            strategy="most_frequent", keep_empty_features=True
                        ),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            make_column_selector(dtype_exclude="number"),
        ),
    ]
    if indicators:
        transforms.append(
            (
                "miss",
                MissingIndicator(features="missing-only", error_on_new=False),
                make_column_selector(),
            )
        )
    return Pipeline(
        [
            ("clean", TrainColumnCleaner()),
            ("prep", ColumnTransformer(transforms)),
            ("clf", model),
        ]
    )


def suggest(trial, name):
    if name == "DT":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
        }
    if name == "RF":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.5]
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]
            ),
        }
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10, log=True),
    }


def tune(name, x, y, seed, out, tag, trials=30):
    inner = list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(x, y)
    )
    assert min(np.bincount(y)) >= 3

    def objective(trial):
        pipe = pipeline(name, seed, True, suggest(trial, name))
        return float(
            cross_val_score(pipe, x, y, cv=inner, scoring="f1_macro", n_jobs=1).mean()
        )

    study = optuna.create_study(
        study_name=tag,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage="sqlite:///" + (out / "optuna.sqlite3").as_posix(),
        load_if_exists=True,
    )
    remaining = max(
        0, trials - len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
    )
    if remaining:
        study.optimize(objective, n_trials=remaining)
    return study


def nadeau_bengio(a, b, ratio=0.25):
    differences = np.asarray(a) - np.asarray(b)
    mean = float(differences.mean())
    variance = float(differences.var(ddof=1))
    denominator = np.sqrt((1 / len(differences) + ratio) * variance)
    if denominator == 0:
        t = 0.0 if mean == 0 else None
        p = 1.0 if mean == 0 else 0.0
    else:
        t = float(mean / denominator)
        p = float(2 * stats.t.sf(abs(t), len(differences) - 1))
    return {
        "mean_difference": mean,
        "t": t,
        "p": p,
        "df": len(differences) - 1,
        "test_train_ratio": ratio,
        "significant_005": p < 0.05,
    }


def summary(scores):
    return {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "folds": list(map(float, values)),
        }
        for key, values in scores.items()
        if key.startswith("test_")
    }


def rank_plot(out, name, labels, values, xlabel):
    idx = np.argsort(values)[-15:]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(np.asarray(labels)[idx], np.asarray(values)[idx], color="#28649b")
    ax.set_xlabel(xlabel)
    return savefig(out, name)


def run(number):
    assert isinstance(number, int) and number > 0
    seed = 42 + number
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out = folder(4)
    raw = read_annealing()
    x = raw.drop(columns="class")
    encoder = LabelEncoder()
    y = encoder.fit_transform(raw["class"])
    tr, te = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed
    )
    xtr = x.iloc[tr]
    xte = x.iloc[te]
    ytr = y[tr]
    yte = y[te]
    cleaner = TrainColumnCleaner().fit(xtr)
    cleaned = cleaner.transform(xtr)
    missing = cleaned.isna().mean().sort_values(ascending=False)
    association = []
    for col in missing[missing > 0].index[:5]:
        table = pd.crosstab(cleaned[col].isna(), ytr).reindex(
            [False, True], fill_value=0
        )
        chi = stats.chi2_contingency(table)
        if np.min(chi.expected_freq) < 5:
            test = stats.fisher_exact(
                table.to_numpy(),
                method=stats.MonteCarloMethod(
                    n_resamples=10000, rng=np.random.default_rng(seed)
                ),
            )
            p = float(test.pvalue)
            method = "Fisher, Monte Carlo 10000"
        else:
            p = float(chi.pvalue)
            method = "Chi-square"
        association.append(
            {
                "feature": col,
                "missing_fraction": float(missing[col]),
                "method": method,
                "p": p,
                "table": table.to_numpy().tolist(),
            }
        )
    rank_plot(
        out,
        "missingness",
        missing.index,
        missing.values * 100,
        "Пропущені значення у Train, %",
    )
    raw["class"].value_counts().sort_index().plot.barh(color="#28649b")
    plt.xlabel("Кількість виробів")
    savefig(out, "class_balance")
    cv = list(
        RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed).split(
            xtr, ytr
        )
    )
    scoring = {"acc": "accuracy", "f1m": "f1_macro", "bacc": "balanced_accuracy"}
    baseline = {}
    raw_scores = {}
    for flag in [False, True]:
        for name in ["DT", "RF", "XGB"]:
            key = ("B" if flag else "A") + " / " + name
            cache = out / (str(seed) + "_" + key.replace(" / ", "_") + "_baseline.json")
            if cache.exists():
                values = json.loads(cache.read_text())
            else:
                scores = cross_validate(
                    pipeline(name, seed, flag),
                    xtr,
                    ytr,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1,
                )
                values = summary(scores)
                cache.write_text(json.dumps(values), encoding="utf-8")
            baseline[key] = values
            raw_scores[key] = values["test_f1m"]["folds"]
            print("BASELINE", key, values["test_f1m"]["mean"], flush=True)
    nested = {}
    final_studies = {}
    models = {}
    best_params = {}
    outer = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(xtr, ytr)
    )
    for name in ["DT", "RF", "XGB"]:
        folds = []
        for fold, (a, b) in enumerate(outer):
            assert min(np.bincount(ytr[a])) >= 3
            study = tune(
                name, xtr.iloc[a], ytr[a], seed, out, f"{seed}_{name}_outer{fold}"
            )
            model = pipeline(name, seed, True, study.best_params).fit(
                xtr.iloc[a], ytr[a]
            )
            pred = model.predict(xtr.iloc[b])
            score = float(f1_score(ytr[b], pred, average="macro"))
            folds.append(score)
            print("NESTED", name, fold, score, flush=True)
        nested[name] = {
            "mean": float(np.mean(folds)),
            "std": float(np.std(folds, ddof=1)),
            "folds": folds,
        }
        study = tune(name, xtr, ytr, seed, out, f"{seed}_{name}_full")
        final_studies[name] = study
        best_params[name] = study.best_params
        nested[name]["ordinary_best_value"] = study.best_value
        models[name] = pipeline(name, seed, True, study.best_params).fit(xtr, ytr)
        with (out / (name + ".pkl")).open("wb") as file:
            cloudpickle.dump(models[name], file)
        optuna.visualization.matplotlib.plot_optimization_history(study)
        savefig(out, "optimization_" + name)
        optuna.visualization.matplotlib.plot_param_importances(
            study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=seed)
        )
        savefig(out, "parameter_importance_" + name)
    comparisons = {
        name + " A-B": nadeau_bengio(
            raw_scores["A / " + name], raw_scores["B / " + name]
        )
        for name in ["DT", "RF", "XGB"]
    }
    top = sorted(
        ["DT", "RF", "XGB"],
        key=lambda n: baseline["B / " + n]["test_f1m"]["mean"],
        reverse=True,
    )[:2]
    comparisons["B " + top[0] + "-" + top[1]] = nadeau_bengio(
        raw_scores["B / " + top[0]], raw_scores["B / " + top[1]]
    )
    # Вибір найкращої моделі за nested CV, до доступу до Test.
    winner = max(nested, key=lambda n: nested[n]["mean"])
    test_metrics = {}
    predictions = {}
    importance = {}
    for name, model in models.items():
        p = model.predict_proba(xte)
        test_metrics[name] = classification(yte, p)
        predictions[name] = p.argmax(1)
        feature_names = model[:-1].get_feature_names_out()
        mdi = model[-1].feature_importances_
        perm = permutation_importance(
            model,
            xte,
            yte,
            n_repeats=10,
            scoring="f1_macro",
            random_state=seed,
            n_jobs=1,
        )
        rank_plot(out, "mdi_" + name, feature_names, mdi, "MDI, після перетворення")
        rank_plot(
            out,
            "permutation_" + name,
            x.columns,
            perm.importances_mean,
            "Зменшення Test macro-F1 після перестановки",
        )
        importance[name] = {
            "mdi": dict(zip(feature_names, map(float, mdi))),
            "permutation_mean": dict(zip(x.columns, map(float, perm.importances_mean))),
            "permutation_std": dict(zip(x.columns, map(float, perm.importances_std))),
        }
    cm = np.array(test_metrics[winner]["confusion_matrix"])
    cm_plot(out, "confusion", cm, encoder.classes_)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    ax.grid(False)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(
                j,
                i,
                f"{norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if norm[i, j] > 0.5 else "black",
            )
    ax.set(
        xticks=range(len(cm)),
        yticks=range(len(cm)),
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_,
        xlabel="Прогноз",
        ylabel="Справжній клас",
    )
    fig.colorbar(im, ax=ax)
    savefig(out, "confusion_normalized")
    mask = np.any(
        np.column_stack(list(predictions.values())) != predictions["DT"][:, None],
        axis=1,
    )
    disagreement_rows = np.flatnonzero(mask)
    explanation_source = "test"
    explain_models = models
    if len(disagreement_rows):
        position = int(disagreement_rows[0])
        sample = xte.iloc[[position]]
        truth = int(yte[position])
        original_row = int(te[position])
    else:
        # Лише out-of-fold передбачення: жодних прогнозів моделі на її Train.
        oof = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(xtr, ytr))
        op = {
            name: cross_val_predict(
                pipeline(name, seed, True, best_params[name]), xtr, ytr, cv=oof
            )
            for name in models
        }
        om = np.any(np.column_stack(list(op.values())) != op["DT"][:, None], axis=1)
        positions = np.flatnonzero(om)
        assert len(positions) > 0, (
            "Жодного розходження навіть в OOF; цей факт потребує окремого пояснення."
        )
        position = int(positions[0])
        a, b = next((a, b) for a, b in oof if position in b)
        sample = xtr.iloc[[position]]
        truth = int(ytr[position])
        original_row = int(tr[position])
        explanation_source = "train OOF"
        explain_models = {
            name: pipeline(name, seed, True, best_params[name]).fit(xtr.iloc[a], ytr[a])
            for name in models
        }
    explanations = {}
    for name, model in explain_models.items():
        transformed = model[:-1].transform(sample)
        features = model[:-1].get_feature_names_out()
        clf = model[-1]
        pred = int(model.predict(sample)[0])
        explainer = shap.TreeExplainer(clf)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = np.stack(values, axis=-1)
        values = np.asarray(values)[0, :, pred]
        indices = np.argsort(abs(values))[-5:]
        base = float(np.asarray(explainer.expected_value)[pred])
        output = (
            float(clf.predict(transformed, output_margin=True)[0, pred])
            if name == "XGB"
            else float(clf.predict_proba(transformed)[0, pred])
        )
        error = abs(base + values.sum() - output)
        assert error < 1e-4, (name, error)
        rank_plot(
            out,
            "shap_" + name,
            features[indices],
            values[indices],
            "SHAP: " + ("логіт" if name == "XGB" else "ймовірність"),
        )
        cleaned_sample = model.named_steps["clean"].transform(sample)
        prep = model.named_steps["prep"]
        imputed = {}
        for branch in ["num", "cat"]:
            cols = next(
                cols for label, _, cols in prep.transformers_ if label == branch
            )
            transformer = prep.named_transformers_[branch]
            imputer = (
                transformer if branch == "num" else transformer.named_steps["impute"]
            )
            for col, value in zip(cols, imputer.statistics_):
                if pd.isna(sample.iloc[0][col]):
                    imputed[col] = value
        explanations[name] = {
            "predicted_class": str(encoder.classes_[pred]),
            "probabilities": list(map(float, model.predict_proba(sample)[0])),
            "expected_value": base,
            "output": output,
            "output_scale": "logit" if name == "XGB" else "probability",
            "additivity_error": float(error),
            "top_contributions": [
                {
                    "feature": features[i],
                    "shap": float(values[i]),
                    "transformed_value": float(transformed[0, i]),
                }
                for i in indices[::-1]
            ],
            "imputed_values": imputed,
        }
    explanation = {
        "source": explanation_source,
        "original_row": original_row,
        "true_class": str(encoder.classes_[truth]),
        "raw_features": {
            k: None if pd.isna(v) else v for k, v in sample.iloc[0].items()
        },
        "models": explanations,
    }
    disagreement_table = [
        {
            "row": int(te[i]),
            "true": str(encoder.classes_[yte[i]]),
            **{name: str(encoder.classes_[predictions[name][i]]) for name in models},
        }
        for i in disagreement_rows
    ]
    pd.DataFrame(disagreement_table).to_csv(out / "disagreements.csv", index=False)
    np.savez(out / "split_indices.npz", train=tr, test=te)
    data = {
        "number": number,
        "seed": seed,
        "source": "UCI Annealing: anneal.data + anneal.test, 898 рядків",
        "shape": list(raw.shape),
        "classes": encoder.classes_.tolist(),
        "class_counts": raw["class"].value_counts().to_dict(),
        "split": {"train": len(tr), "test": len(te)},
        "train_class_counts": np.bincount(ytr).tolist(),
        "removed_train_constant_or_empty": cleaner.removed_,
        "missingness_train": missing.to_dict(),
        "missing_association": association,
        "baseline": baseline,
        "nested": nested,
        "best_params": best_params,
        "statistical_tests": comparisons,
        "winner_nested_cv": winner,
        "test": test_metrics,
        "classification_report": classification_report(
            yte,
            predictions[winner],
            target_names=encoder.classes_,
            zero_division=0,
            output_dict=True,
        ),
        "importance": importance,
        "disagreement_count": len(disagreement_rows),
        "disagreements": disagreement_table,
        "local_explanation": explanation,
        "trials_per_study": 30,
        "outer_folds": 5,
        "inner_folds": 3,
        "repeated_cv_folds": 15,
    }
    dump(out, data)
    print("LAB 4 COMPLETE", winner, test_metrics[winner]["macro_f1"], flush=True)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int, required=True)
    run(parser.parse_args().number)
