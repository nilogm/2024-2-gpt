import warnings

warnings.filterwarnings("ignore")

import seaborn as sns
import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def set_legend(fig, ax: Axes, title: str):
    legend_configs = dict(
        loc="upper center",
        fontsize=12,
        title_fontsize=12,
    )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), ncols=5, **legend_configs, title=title)


answer_metrics = {
    # "rouge_score_precision": "ROUGE-L Precision",
    "rouge_score_recall": "ROUGE-L Recall",
    # "rouge_score_f1": "ROUGE-L F1",
    "semantic_similarity_score": "Similaridade Semântica",
    "bert_precision_score": "BERTScore Precision",
}

retrieval_metrics = {
    "date_retrieved": "Busca por Data",
    "date_retrieved_right_date": "Busca por Data (c/ marcador)",
    "semantic_retrieved": "Busca por Semântica",
    "semantic_or_date_retrieved": "Busca por Semântica ou Data",
    "relevant_retrieved": "Análise de Relevância",
}

all_metrics = [i for i in answer_metrics.keys()] + [i for i in retrieval_metrics.keys()]


def read_folder(path: str) -> pd.DataFrame:
    df = None
    for file in os.listdir(path):
        df_ = pd.read_csv(os.path.join(path, file))
        generative, encoder, top_k, summarizer = file.removesuffix(".csv").split("__")[:4]
        df_["generative_model"] = generative
        df_["encoder_model"] = encoder
        df_["top_k"] = top_k
        df_["summarizer"] = summarizer
        df = df_ if df is None else pd.concat([df, df_], ignore_index=True)
    return df


def display_scores(df: pd.DataFrame, x: str, hue: str, x_name: str = None, x_labels: dict = None, hue_name: str = None, hue_labels: dict = None, img_name: str = None):
    if x not in df.columns or hue not in df.columns:
        raise ValueError("'x' and 'hue' both have to be valid columns in the DataFrame")

    df["x"] = df[x]
    df["y"] = df[hue]

    if x_labels is not None:
        df["x"] = df["x"].map(x_labels)
    if hue_labels is not None:
        df["y"] = df["y"].map(hue_labels)

    def setup_figure(fig, ax: np.ndarray, metrics: dict, error_bar=True):
        for (metric, title), a in zip(metrics.items(), ax.flatten()):
            a: Axes = a
            a.grid()
            a.set_title(title)
            
            if error_bar:
                sns.barplot(data=df, y=metric, x="x", ax=a, palette="tab10", hue="y")
            else:
                sns.barplot(data=df, y=metric, x="x", ax=a, palette="tab10", hue="y", errorbar=None)

            a.set_ylim(0, 100)
            a.set_yticks(np.linspace(0, 100, 11))
            a.set_yticklabels(np.linspace(0, 100, 11, dtype=int))

            a.set_xlabel(x_name if x_name is not None else x)
            a.set_ylabel("")

        if ax.flatten()[0].get_legend():
            set_legend(fig, ax.flatten()[0], hue_name if hue_name is not None else hue)

        [i.get_legend().remove() for i in ax.flatten() if i.get_legend() is not None]

    fig, ax = plt.subplots(1, 3, figsize=[15, 5], tight_layout=True)
    setup_figure(fig, ax, answer_metrics)
    plt.savefig(f"answer_{img_name}.svg", bbox_inches="tight")

    fig, ax = plt.subplots(1, 5, figsize=[25, 5], tight_layout=True)
    setup_figure(fig, ax, retrieval_metrics, False)
    plt.savefig(f"retrieval_{img_name}.svg", bbox_inches="tight")

    print(df.groupby([x, hue]).agg({i: ["mean", "std"] for i in all_metrics}))
