import faiss.swigfaiss
import pandas as pd
import numpy as np
import traceback
import datetime
import faiss
import time
from faiss import read_index
from sentence_transformers import SentenceTransformer
from gpt_2024_2.model.summarizer import get_theme


def _get_files(file: dict):
    try:
        index: faiss.swigfaiss.IndexIDMap = read_index(file["index"])
        chunks = pd.read_csv(file["corpus"])
        return index, chunks
    except Exception:
        raise FileNotFoundError(f"One of the following necessary files were not found: '{file['corpus']}' or '{file['index']}'")


class SentenceRetriever:
    def __init__(self, encoder_model_id: str, top_k: int = 3, borders: int = 1, device: int = 0, rand_sample_percent: float = 0.01):
        self.encoder_model_id = encoder_model_id
        self.device = device
        self.top_k = top_k
        self.borders = borders
        self.rand_sample_percent = rand_sample_percent if rand_sample_percent <= 1.0 else 1.0

    def load(self):
        self.model = SentenceTransformer(self.encoder_model_id, trust_remote_code=True, device=(f"cuda:{self.device}" if isinstance(self.device, int) else None))

    def unload(self):
        del self.model
        self.model = None

    def load_database(self, files: dict):
        self.data_chunks, self.df_data = _get_files(files)

    def fetch_conversation(self, df_corpus: pd.DataFrame, dataframe_idx, dates: dict) -> str:
        info = df_corpus.iloc[dataframe_idx]

        date = pd.to_datetime(info["date"], dayfirst=True)
        include = len(dates["dates"]) == 0

        for i in dates["dates"]:
            start_date = pd.to_datetime(i["start"], dayfirst=True)
            if "end" in i:
                end_date = pd.to_datetime(i["end"], dayfirst=True)
                if start_date <= date <= end_date:
                    include = True
            else:
                if start_date == date:
                    include = True

        if not include:
            return ""

        text = f"Data: {info['date']}, {info['weekday']}\n" + "\n".join([f" - {msg}" for msg in eval(info["dialog"])])
        return text

    def is_from_same_conversation(self, i: int, j: int) -> bool:
        return self.df_data.iloc[i]["id"] == self.df_data.iloc[j]["id"]

    def retrieve(self, query, dates: dict, verbose: bool = False):
        t = time.time()

        # TODO: com base nas datas, filtrar o index para buscar o conteúdo dentre estes itens.

        # TODO: talvez aqui seria bom implementar algo que ao detectar índices próximos com mesmo "id" ele retorna os border antes do menor índice e border após o maior índice.
        def check_closeness(idx: list[int]):
            for i in range(len(idx)):
                for j in range(0, i):
                    # if abs(idx[i] - idx[j]) <= self.borders and self.is_from_same_conversation(idx[i], idx[j]):
                    if self.is_from_same_conversation(idx[i], idx[j]):
                        idx[i] = idx[j]

            return idx

        def search_in_index(query_vector):
            search_items = self.data_chunks.search(query_vector, (2 * self.borders + 1) * self.top_k)
            top_k_ids_text = search_items[1][0]  # retorna [(dist, ), (ids, )]

            top_k_ids_text = check_closeness(top_k_ids_text)

            unique, index = np.unique(top_k_ids_text, return_index=True)
            top_k_ids = unique[index.argsort()]

            return [self.fetch_conversation(self.df_data, idx, dates) for idx in top_k_ids[: self.top_k]]

        # olhar também aleatoriamente uma % muito pequena da nossa memória
        # Não melhora significativamente o resultado e fica grande demais pra ler e validar
        def get_rand_sample():
            if self.rand_sample_percent <= 0.0:
                return []
            total_items = len(self.df_data) // 2
            return [self.fetch_conversation(self.df_data, idx) for idx in np.random.choice(total_items, size=int(total_items * self.rand_sample_percent), replace=False)]

        try:
            # mantendo o padrão de busca em vizinhança do tema/query
            encoded_query = self.model.encode([get_theme(query, "bart")])
            results = search_in_index(encoded_query)

        except Exception as e:
            print("Error: ", str(e))
            traceback.print_exc()
            print("An error occured when retrieving memories!")

        et = time.time() - t
        if verbose:
            print(">>>> Results in Total Time: {}".format(et))

        return results, et

    # TODO: implementar um retrieve que busca por data
    # a ideia é que podemos também procurar conversas relacionadas à data. Ex.: "Do que estávamos falando ontem?"
    def retrieve_from_date(self, query: str, date: str):
        self.df_data[self.df_data["date"] == date]
        raise NotImplementedError("A função de busca por data ainda não foi implementada!")
