import pandas as pd
import numpy as np
import os
import time
import faiss
from sentence_transformers import SentenceTransformer


def build_chunks(path: str, destination_path: str, key: str, verbose: bool = False):
    corpora = pd.DataFrame(columns=["id", "dialog", "act", "emotion"])

    if verbose:
        print(f"Building memories from conversations...")

    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, filename))
            corpora = pd.concat([corpora, pd.DataFrame({"id": df["id"], "dialog": df["dialog"]})], ignore_index=True)

    # TODO: implementar isso
    #corpora["date"] = None

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

    encoded_data = encoder_model.encode(df["dialog"].tolist(), show_progress_bar=verbose)
    encoded_data = np.asarray(encoded_data.astype("float32"))
    index = faiss.IndexIDMap(faiss.IndexFlatIP(encoder_model_dimensions))
    index.add_with_ids(encoded_data, np.array(range(len(df))))

    faiss.write_index(index, destination)

    if verbose:
        print(">>>> Created embeddings in total time: {}".format(time.time() - t))
        print(f"Index saved to '{destination}'")
