import faiss.swigfaiss
import pandas as pd
import numpy as np
import traceback
import faiss
import time
from faiss import read_index
from sentence_transformers import SentenceTransformer
from gpt_2024_2.model.summarizer import get_theme, SUMMARIZERS


def _get_files(file: dict):
    try:
        index: faiss.swigfaiss.IndexIDMap = read_index(file["index"])
        chunks = pd.read_csv(file["corpus"])
        chunks["date"] = pd.to_datetime(chunks["date"], dayfirst=True).dt.date
        return index, chunks
    except Exception:
        raise FileNotFoundError(f"One of the following necessary files were not found: '{file['corpus']}' or '{file['index']}'")


class SentenceRetriever:
    def __init__(self, encoder_model_id: str, top_k: int = 3, summarizer: str = "default", rand_sample_percent: float = 0.01, device: int = 0, **kwargs):
        self.encoder_model_id = encoder_model_id
        self.device = device
        self.top_k = top_k
        self.summarizer = summarizer if summarizer in SUMMARIZERS.keys() else "default"
        self.rand_sample_percent = rand_sample_percent if rand_sample_percent <= 1.0 else 1.0
        self.model = None

    def load(self):
        if self.model is None:
            self.model = SentenceTransformer(self.encoder_model_id, trust_remote_code=True, device=(f"cuda:{self.device}" if isinstance(self.device, int) else None))

    def unload(self):
        del self.model
        self.model = None

    def load_database(self, files: dict):
        self.data_chunks, self.df_data = _get_files(files)

    def fetch_conversation(self, dataframe_idx, dates: dict) -> str:
        info = self.df_data.iloc[dataframe_idx]
        include = len(dates["dates"]) == 0

        for i in dates["dates"]:
            start_date = pd.to_datetime(i["start"], dayfirst=False).date()
            if "end" in i:
                end_date = pd.to_datetime(i["end"], dayfirst=False).date()
                if start_date <= info["date"] <= end_date:
                    include = True
            else:
                if start_date == info["date"]:
                    include = True

        if not include:
            return ""

        text = f"Date: {info['date']}, {info['weekday']}\n" + "\n".join([f" > {'User' if i %2 == 0 else 'You'}: {msg}" for i, msg in enumerate(eval(info["dialog"]))])
        return text

    def filter_ids(self, dates: dict) -> str:
        include_idx = []

        for i in dates["dates"]:
            start_date = pd.to_datetime(i["start"]).date()
            if "end" in i:
                end_date = pd.to_datetime(i["end"]).date()
                include_idx.extend(self.df_data[(start_date <= self.df_data["date"]) & (self.df_data["date"] <= end_date)].index)
            else:
                include_idx.extend(self.df_data[(start_date == self.df_data["date"])].index)

        return include_idx

    # olhar também aleatoriamente uma % muito pequena da nossa memória
    # Não melhora significativamente o resultado e fica grande demais pra ler e validar
    def get_rand_sample(self):
        if self.rand_sample_percent <= 0.0:
            return []

        total_items = len(self.df_data) // 2
        return [self.fetch_conversation(self.df_data, idx) for idx in np.random.choice(total_items, size=int(total_items * self.rand_sample_percent), replace=False)]

    def retrieve(self, query, dates: dict, verbose: bool = False):
        t = time.time()

        try:
            # mantendo o padrão de busca em vizinhança do tema/query
            encoded_query = self.model.encode([get_theme(query, self.summarizer)])

            search_items = self.data_chunks.search(encoded_query, self.top_k)
            top_k_ids_text = search_items[1][0]  # retorna [(dist, ), (ids, )]

            unique, index = np.unique(top_k_ids_text, return_index=True)
            top_k_ids = unique[index.argsort()]

            results = [(idx, self.fetch_conversation(idx, dates)) for idx in top_k_ids[: self.top_k]]

        except Exception as e:
            print("Error: ", str(e))
            traceback.print_exc()
            print("An error occured when retrieving memories!")

        et = time.time() - t
        if verbose:
            print(">>>> Results in Total Time: {}".format(et))

        return results, et

    def retrieve_by_date(self, dates: dict, verbose: bool = True):
        t = time.time()

        try:
            all_idx = self.filter_ids(dates)
            results = [(idx, self.fetch_conversation(idx, dates)) for idx in all_idx]

        except Exception as e:
            print("Error: ", str(e))
            traceback.print_exc()
            print("An error occured when retrieving memories!")

        et = time.time() - t
        if verbose:
            print(">>>> Results in Total Time: {}".format(et))

        return results, et
