import faiss.swigfaiss
import pandas as pd
import numpy as np
import traceback
import faiss
import time
from faiss import read_index
from sentence_transformers import SentenceTransformer


def _get_files(corpus: str, index: str):
    try:
        index: faiss.swigfaiss.IndexIDMap = read_index(index)
        chunks = pd.read_csv(corpus)
        chunks["date"] = pd.to_datetime(chunks["date"], dayfirst=True).dt.date
        return index, chunks
    except Exception:
        raise FileNotFoundError(f"One of the following necessary files were not found: '{corpus}' or '{index}'")


class SentenceRetriever:
    def __init__(self, encoder_model_id: str, top_k: int = 3, device: int = 0, **kwargs):
        self.encoder_model_id = encoder_model_id
        self.device = device
        self.top_k = top_k
        self.model = None

    def load(self):
        if self.model is None:
            self.model = SentenceTransformer(self.encoder_model_id, trust_remote_code=True, device=(f"cuda:{self.device}" if isinstance(self.device, int) else None))

    def unload(self):
        del self.model
        self.model = None

    def load_database(self, corpus: str, index: str):
        self.data_chunks, self.df_data = _get_files(corpus=corpus, index=index)

    def fetch_conversation(self, dataframe_idx, dates: dict) -> str:
        info = self.df_data.iloc[dataframe_idx]

        if len(dates["dates"]) > 0:
            for i in dates["dates"]:
                start_date = pd.to_datetime(i["start"], dayfirst=False).date()
                if "end" in i:
                    end_date = pd.to_datetime(i["end"], dayfirst=False).date()
                    if start_date <= info["date"] <= end_date:
                        break
                else:
                    if start_date == info["date"]:
                        break
            else:
                return ""

        text = f"Date: {info['date']}, {info['weekday']}\n" + "\n".join([f" > {'User' if i %2 == 0 else 'You'}: {msg}" for i, msg in enumerate(eval(info["dialog"]))])
        return text

    def filter_ids(self, dates: dict) -> str:
        include_idx = []

        for i in dates["dates"]:
            start_date = pd.to_datetime(i["start"], dayfirst=False).date()
            if "end" in i:
                end_date = pd.to_datetime(i["end"], dayfirst=False).date()
                include_idx.extend(self.df_data[(start_date <= self.df_data["date"]) & (self.df_data["date"] <= end_date)].index)
            else:
                include_idx.extend(self.df_data[(start_date == self.df_data["date"])].index)

        return include_idx

    def retrieve(self, query, dates: dict):
        t = time.time()

        try:
            # mantendo o padrão de busca em vizinhança do tema/query
            encoded_query = self.model.encode([query])

            search_items = self.data_chunks.search(encoded_query, self.top_k * 3)
            top_k_ids_text = search_items[1][0]  # retorna [(dist, ), (ids, )]

            unique, index = np.unique(top_k_ids_text, return_index=True)
            top_k_ids = unique[index.argsort()]

            results = []
            for i in top_k_ids:
                conv = self.fetch_conversation(i, dates)
                if conv != "":
                    results.append((i, conv))
                if len(results) >= self.top_k:
                    break

        except Exception as e:
            print("Error: ", str(e))
            traceback.print_exc()
            print("An error occured when retrieving memories!")

        et = time.time() - t

        return results, et

    def retrieve_by_date(self, dates: dict):
        t = time.time()

        try:
            all_idx = self.filter_ids(dates)
            results = [(idx, self.fetch_conversation(idx, dates)) for idx in all_idx]

        except Exception as e:
            print("Error: ", str(e))
            traceback.print_exc()
            print("An error occured when retrieving memories!")

        et = time.time() - t

        return results, et
