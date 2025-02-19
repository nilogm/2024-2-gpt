import faiss.swigfaiss
import pandas as pd
import numpy as np
import traceback
import faiss
import time
from faiss import read_index
from sentence_transformers import SentenceTransformer


def _get_files(file: dict):
    try:
        index: faiss.swigfaiss.IndexIDMap = read_index(file["index"])
        chunks = pd.read_csv(file["corpus"])
        return index, chunks
    except Exception:
        raise FileNotFoundError(f"One of the following necessary files were not found: '{file['corpus']}' or '{file['index']}'")


class SentenceRetriever:
    def __init__(self, encoder_model_id: str, top_k: int = 3, borders: int = 1, device: int = 0):
        self.encoder_model_id = encoder_model_id
        self.device = device
        self.top_k = top_k
        self.borders = borders

    def load(self):
        self.model = SentenceTransformer(self.encoder_model_id, trust_remote_code=True, device=(f"cuda:{self.device}" if isinstance(self.device, int) else None))

    def unload(self):
        del self.model
        self.model = None

    def load_database(self, files: dict):
        self.data_chunks, self.df_data = _get_files(files)

    def fetch_conversation(self, df_corpus: pd.DataFrame, dataframe_idx) -> str:
        if self.borders == 0:
            info = df_corpus.iloc[dataframe_idx]
            return info["message"]

        info = df_corpus.iloc[dataframe_idx - self.borders : dataframe_idx + self.borders + 1]
        info = info[info["conversation_id"] == df_corpus.iloc[dataframe_idx]["conversation_id"]]

        text = "\n".join([f" - {msg}" for msg in info["message"]])
        return text

    def is_from_same_conversation(self, i: int, j: int) -> bool:
        return self.df_data.iloc[i]["conversation_id"] == self.df_data.iloc[j]["conversation_id"]

    def retrieve(self, query, verbose: bool = False):
        t = time.time()
        query_vector = self.model.encode([query])

        # TODO: talvez aqui seria bom implementar algo que ao detectar índices próximos com mesmo "conversation_id" ele retorna os border antes do menor índice e border após o maior índice.
        def check_closeness(idx: list[int]):
            for i in range(len(idx)):
                for j in range(0, i):
                    if abs(idx[i] - idx[j]) <= self.borders and self.is_from_same_conversation(idx[i], idx[j]):
                        idx[i] = idx[j]

            return idx

        def search_in_index():
            search_items = self.data_chunks.search(query_vector, (2 * self.borders + 1) * self.top_k)
            top_k_ids_text = search_items[1][0]  # retorna [(dist, ), (ids, )]

            top_k_ids_text = check_closeness(top_k_ids_text)

            unique, index = np.unique(top_k_ids_text, return_index=True)
            top_k_ids = unique[index.argsort()]

            return [self.fetch_conversation(self.df_data, idx) for idx in top_k_ids[: self.top_k]]

        try:
            if self.df_data is not None:
                results = search_in_index()

        except Exception as e:
            print("Error: ", str(e))
            traceback.print_exc()
            print("An error occured when retrieving contexts!")

        et = time.time() - t
        if verbose:
            print(">>>> Results in Total Time: {}".format(et))

        return results, et

    # TODO: implementar um retrieve que busca por data
    # a ideia é que podemos também procurar conversas relacionadas à data. Ex.: "Do que estávamos falando ontem?"
    def retrieve_from_date(self):
        raise NotImplementedError("A função de busca por data ainda não foi implementada!")
