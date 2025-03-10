import pandas as pd
import numpy as np
import os
import re
import time
import faiss
import datetime
from sentence_transformers import SentenceTransformer


def build_chunks(path: str, destination_path: str, key: str, verbose: bool = False):
    corpora = pd.DataFrame(columns=["id", "dialog", "date", "weekday"])

    if verbose:
        print(f"Building memories from conversations...")

    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, filename))
            df = modify_df(df)
            df.to_csv("memories_w_dates.csv", index=False)
            corpora = pd.concat([corpora, pd.DataFrame({"id": df["conversation_id"], "dialog": df["dialog"], "date": df["date"], "weekday": df["weekday"]})], ignore_index=True)[:1000]

    if verbose:
        print(corpora.tail())
        print("Done building memories from conversations.")

    corpora.reset_index(inplace=True)
    corpora.to_csv(f"{destination_path}/{key}.csv", index=False, escapechar="\\")

    if verbose:
        print(f"Conversation memory saved in '{destination_path}'")


def build_index(source: str, destination: str, encoder_model: SentenceTransformer, encoder_model_dimensions: int = 1024, verbose: bool = False):
    if verbose:
        print(f"Creating index from {source}...")

    t = time.time()
    df = pd.read_csv(source)
    df["dialog"] = df["dialog"].apply(lambda x: eval(x))
    df = df.explode("dialog")

    encoded_data = encoder_model.encode(df["dialog"].tolist(), show_progress_bar=verbose)
    encoded_data = np.asarray(encoded_data.astype("float32"))
    index = faiss.IndexIDMap(faiss.IndexFlatIP(encoder_model_dimensions))
    index.add_with_ids(encoded_data, df["id"].to_numpy())

    faiss.write_index(index, destination)

    if verbose:
        print(">>>> Created embeddings in total time: {}".format(time.time() - t))
        print(f"Index saved to '{destination}'")


def modify_df(df: pd.DataFrame):
    df["dialog"] = df["dialog"].apply(lambda x: [j.strip() for i in re.findall(r"'([^']*)'|\"([^\"]*)\"", x) for j in i if len(j.strip()) > 0])
    start_date = datetime.date(2025, 3, 3)
    df["date"] = df.apply(lambda x: (start_date - datetime.timedelta(days=x["conversation_id"])).strftime("%d/%m/%Y"), axis=1)
    df["weekday"] = pd.to_datetime(df["date"], format="%d/%m/%Y").dt.day_name()
    return df
