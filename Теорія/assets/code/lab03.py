"""Практична 3: FastText та агрегація документів, без навчання на Test."""

import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from gensim.models import FastText
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import (
    roc_auc_score,
    silhouette_score,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)
from common import DATA, folder, savefig, dump, cm_plot, stable_hash


def preprocess(texts):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return [
        [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]
        for doc in nlp.pipe(texts, batch_size=64)
    ]


class DocumentVectors:
    def fit(self, tokens):
        self.ft = FastText(
            sentences=tokens,
            vector_size=100,
            window=5,
            min_count=2,
            sg=1,
            workers=1,
            seed=42,
            hashfxn=stable_hash,
            bucket=100000,
            epochs=20,
        )
        self.tfidf = TfidfVectorizer(
            analyzer=lambda x: x, lowercase=False, token_pattern=None, norm=None
        )
        self.tfidf.fit(tokens)
        self.idf = dict(zip(self.tfidf.get_feature_names_out(), self.tfidf.idf_))
        return self

    def transform(self, corpus, weighted=False):
        rows = []
        empty = 0
        for tokens in corpus:
            if not tokens:
                rows.append(np.zeros(100))
                empty += 1
                continue
            counts = Counter(tokens)
            terms = list(counts)
            vectors = np.array([self.ft.wv[t] for t in terms])
            # Кожне унікальне слово один раз: частота вже входить у TF-IDF.
            weights = np.array(
                [counts[t] * (self.idf.get(t, 0) if weighted else 1.0) for t in terms]
            )
            if weights.sum() == 0:
                weights = np.array([counts[t] for t in terms])
            rows.append(np.average(vectors, weights=weights, axis=0))
        return np.asarray(rows), empty


def evaluate(tokens, y, train, test, out, prefix, make_plots=True):
    vec = DocumentVectors().fit([tokens[i] for i in train])
    scores = {}
    spaces = {}
    probabilities = {}
    silhouettes = {}
    for method, weighted in [("Mean", False), ("TF-IDF", True)]:
        x, empty = vec.transform([tokens[i] for i in train], weighted)
        xt, empty_test = vec.transform([tokens[i] for i in test], weighted)
        models = {
            "SVM": SVC(kernel="rbf", probability=True, random_state=42),
            "LogReg": LogisticRegression(max_iter=2000, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
        }
        for name, model in models.items():
            pipe = make_pipeline(StandardScaler(), model).fit(x, y[train])
            p = pipe.predict_proba(xt)[:, 1]
            pred = pipe.predict(xt)
            score = {
                "accuracy": float(accuracy_score(y[test], pred)),
                "macro_f1": float(f1_score(y[test], pred, average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
                "roc_auc": float(roc_auc_score(y[test], p)),
            }
            key = method + " / " + name
            scores[key] = score
            probabilities[key] = p
            if make_plots:
                from sklearn.metrics import confusion_matrix

                cm_plot(
                    out,
                    "confusion_" + method.replace("-", "") + "_" + name,
                    confusion_matrix(y[test], pred),
                    ["Human", "AI"],
                )
        spaces[method] = xt
        silhouettes[method] = float(silhouette_score(xt, y[test]))
    if make_plots:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, (method, space) in zip(axes, spaces.items()):
            xy = TSNE(
                n_components=2,
                perplexity=min(30, len(test) // 4),
                random_state=42,
                init="pca",
                learning_rate="auto",
            ).fit_transform(space)
            for cls, label in [(0, "Human"), (1, "AI")]:
                ax.scatter(*xy[y[test] == cls].T, s=15, alpha=0.65, label=label)
            ax.set(title=method, xlabel="t-SNE 1", ylabel="t-SNE 2")
            ax.legend()
        savefig(out, "document_spaces")
        # Важливі слова визначені лише на Train, не по Test.
        tf = vec.tfidf.transform([tokens[i] for i in train])
        means = np.asarray(tf[y[train] == 1].mean(0)).ravel()
        names = vec.tfidf.get_feature_names_out()
        order = np.argsort(means)[-15:]
        plt.barh(names[order], means[order], color="#28649b")
        plt.xlabel("Середній TF-IDF у тренувальних текстах AI")
        savefig(out, "ai_vocabulary")
        vec.ft.save(str(out / "fasttext.model"))
        np.savez(
            out / "test_vectors.npz",
            mean=spaces["Mean"],
            weighted=spaces["TF-IDF"],
            y=y[test],
            row_indices=test,
        )
        top_words = [
            {"word": names[i], "mean_tfidf": float(means[i])} for i in order[::-1]
        ]
    else:
        top_words = []
    return scores, silhouettes, top_words


def run():
    out = folder(3)
    raw = pd.read_csv(DATA / "ai_vs_human_text_2026.csv")
    df = raw.dropna(subset=["text_content", "label"]).copy()
    df["canonical"] = df.text_content.map(
        lambda x: re.sub(r"\s+", " ", str(x).strip().lower())
    )
    conflicts = int(df.groupby("canonical").label.nunique().gt(1).sum())
    assert conflicts == 0
    duplicate_count = int(df.duplicated("canonical").sum())
    df = df.drop_duplicates("canonical")
    short_count = int(df.canonical.str.split().str.len().lt(5).sum())
    df = df[df.canonical.str.split().str.len() >= 5].reset_index(drop=True)
    labels = sorted(df.label.unique())
    assert set(labels) == {"ai", "human"}, labels
    y = (df.label == "ai").to_numpy(int)
    ids = np.arange(len(df))
    train, test = train_test_split(ids, test_size=0.2, stratify=y, random_state=42)
    tokens = preprocess(df.text_content.tolist())
    assert all(tokens)
    results, silhouettes, top_words = evaluate(tokens, y, train, test, out, "main")
    # Перевірка перенесення на нові теми: забороняє однакову тему в Train/Test.
    groups = df.topic_hint.fillna("unknown").to_numpy()
    gtrain, gtest = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            ids, y, groups
        )
    )
    assert not set(groups[gtrain]) & set(groups[gtest])
    group_results, group_silhouette, _ = evaluate(
        tokens, y, gtrain, gtest, out, "group", False
    )
    lengths = df.text_content.str.split().str.len()
    for cls, label in [(0, "Human"), (1, "AI")]:
        plt.hist(lengths[y == cls], bins=30, alpha=0.55, label=label)
    plt.xlabel("Кількість слів")
    plt.ylabel("Кількість текстів")
    plt.legend()
    savefig(out, "text_lengths")
    frame = pd.DataFrame(results).T[["macro_f1", "roc_auc"]]
    frame.plot.barh(figsize=(9, 4.5))
    plt.xlim(0, 1)
    plt.xlabel("Оцінка на Test")
    savefig(out, "model_comparison")
    (out / "preprocessed_tokens.json").write_text(
        __import__("json").dumps(tokens), encoding="utf-8"
    )
    df[["text_id", "label", "topic_hint", "generation_method"]].to_csv(
        out / "sample_metadata.csv", index=False
    )
    data = {
        "source": "alitaqishah/ai-vs-human-text-classification-dataset-2026",
        "raw_rows": len(raw),
        "rows": len(df),
        "duplicates_removed": duplicate_count,
        "short_removed": short_count,
        "conflicting_labels": conflicts,
        "class_counts": df.label.value_counts().to_dict(),
        "generation_methods": df.generation_method.value_counts().to_dict(),
        "source_models": df.source_model.value_counts().to_dict(),
        "topics": df.topic_hint.nunique(),
        "split": {"train": len(train), "test": len(test)},
        "embedding": {
            "dimension": 100,
            "window": 5,
            "min_count": 2,
            "sg": 1,
            "workers": 1,
            "seed": 42,
            "epochs": 20,
            "bucket": 100000,
        },
        "models": results,
        "silhouette_100d": silhouettes,
        "top_ai_words": top_words,
        "group_holdout": {
            "train": len(gtrain),
            "test": len(gtest),
            "train_topics": len(set(groups[gtrain])),
            "test_topics": len(set(groups[gtest])),
            "overlapping_topics": 0,
            "models": group_results,
        },
        "limitation": "Розмітка є заявою автора датасету. Метадані generation_method містять шаблонні способи формування текстів; оцінки не доводять здатність визначати авторство довільного тексту.",
    }
    np.savez(
        out / "split_indices.npz",
        train=train,
        test=test,
        group_train=gtrain,
        group_test=gtest,
    )
    dump(out, data)
    print("LAB 3 COMPLETE", results, flush=True)
    return data


if __name__ == "__main__":
    run()
