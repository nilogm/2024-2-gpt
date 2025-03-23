import pandas as pd
import numpy as np
import os
import re
import time
import faiss
import datetime
from sentence_transformers import SentenceTransformer
from gpt_2024_2.model.summarizer import get_theme, SUMMARIZERS


def build_chunks(path: str, destination_path: str, end_date: datetime.date = datetime.date(2025, 3, 3), verbose: bool = False):
    corpora = pd.DataFrame(columns=["id", "dialog", "date", "weekday"])

    if verbose:
        print(f"Building memories from conversations...")

    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, filename))
            df = modify_df(df, end_date=end_date)
            corpora = pd.concat([corpora, pd.DataFrame({"id": df["conversation_id"], "dialog": df["dialog"], "date": df["date"], "weekday": df["weekday"]})], ignore_index=True)

    if verbose:
        print(corpora.tail())
        print("Done building memories from conversations.")

    corpora.reset_index(inplace=True)
    corpora.to_csv(f"{destination_path}/memories.csv", index=False, escapechar="\\")

    if verbose:
        print(f"Conversation memory saved in '{destination_path}'")


def build_index(source: str, destination: str, encoder_model: SentenceTransformer, encoder_model_dimensions: int = 1024, summarizer: str = "default", verbose: bool = False):
    if verbose:
        print(f"Creating index from {source}...")

    t = time.time()
    df = pd.read_csv(source)
    df["dialog"] = df["dialog"].apply(lambda x: eval(x))

    summarizer_ = summarizer if summarizer in SUMMARIZERS.keys() else "default"
    themes = df["dialog"].apply(lambda x: get_theme("\n".join([f" > {'User' if i %2 == 0 else 'You'}: {msg}" for i, msg in enumerate(x)]), summarizer_)).tolist()

    encoded_data = encoder_model.encode(themes, show_progress_bar=verbose)
    encoded_data = np.asarray(encoded_data.astype("float32"))
    index = faiss.IndexIDMap(faiss.IndexFlatIP(encoder_model_dimensions))
    index.add_with_ids(encoded_data, df["id"].to_numpy())

    faiss.write_index(index, destination)

    if verbose:
        print(">>>> Created embeddings in total time: {}".format(time.time() - t))
        print(f"Index saved to '{destination}'")


def modify_df(df: pd.DataFrame, end_date: datetime.date = datetime.date(2025, 3, 3)):
    df["dialog"] = df["dialog"].apply(lambda x: [j.strip() for i in re.findall(r"'([^']*)'|\"([^\"]*)\"", x) for j in i if len(j.strip()) > 0])
    df["date"] = df.apply(lambda x: (end_date - datetime.timedelta(days=x["conversation_id"])).strftime("%d/%m/%Y"), axis=1)
    df["weekday"] = pd.to_datetime(df["date"], format="%d/%m/%Y").dt.day_name()
    return df
