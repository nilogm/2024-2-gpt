import faiss.swigfaiss
import pandas as pd
import numpy as np
import traceback
import faiss
import time
from faiss import read_index
from sentence_transformers import SentenceTransformer
import yake
from rake_nltk import Rake
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
import nltk

def download_punkt():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

def _get_files(file: dict):
    try:
        index: faiss.swigfaiss.IndexIDMap = read_index(file["index"])
        chunks = pd.read_csv(file["corpus"])
        return index, chunks
    except Exception:
        raise FileNotFoundError(f"One of the following necessary files were not found: '{file['corpus']}' or '{file['index']}'")


class SentenceRetriever:
    def __init__(self, encoder_model_id: str, top_k: int = 3, borders: int = 1, device: int = 0, rand_sample_percent : float = 0.01):
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

    def fetch_conversation(self, df_corpus: pd.DataFrame, dataframe_idx) -> str:
        if self.borders == 0:
            info = df_corpus.iloc[dataframe_idx]
            return info["dialog"]

        info = df_corpus.iloc[dataframe_idx - self.borders : dataframe_idx + self.borders + 1]
        info = info[info["id"] == df_corpus.iloc[dataframe_idx]["id"]]

        text = "\n".join([f" - {msg}" for msg in info["dialog"]])
        return text

    def is_from_same_conversation(self, i: int, j: int) -> bool:
        return self.df_data.iloc[i]["id"] == self.df_data.iloc[j]["id"]

    def retrieve(self, query, verbose: bool = False):
        t = time.time()
        
        # TODO: talvez aqui seria bom implementar algo que ao detectar índices próximos com mesmo "id" ele retorna os border antes do menor índice e border após o maior índice.
        def check_closeness(idx: list[int]):
            for i in range(len(idx)):
                for j in range(0, i):
                    if abs(idx[i] - idx[j]) <= self.borders and self.is_from_same_conversation(idx[i], idx[j]):
                        idx[i] = idx[j]

            return idx

        def search_in_index(query_vector):
            search_items = self.data_chunks.search(query_vector, (2 * self.borders + 1) * self.top_k)
            top_k_ids_text = search_items[1][0]  # retorna [(dist, ), (ids, )]

            top_k_ids_text = check_closeness(top_k_ids_text)

            unique, index = np.unique(top_k_ids_text, return_index=True)
            top_k_ids = unique[index.argsort()]

            return [self.fetch_conversation(self.df_data, idx) for idx in top_k_ids[: self.top_k]]
        
        # default -> theme_of_query = query
        def get_theme_of_query(option='default'):
            
            #Todas as alternativas abaixo possuem costumização de parâmetros que podem ser úteis
            
            # pega palavras chaves usando yake, solução simples e de baixo custo computacional
            def theme_from_yake():
                 # somente top=3 pq o texto é pode ser muito pequeno
                extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=3)
                return " ".join([ keyword for keyword, _ in extractor.extract_keywords(query)])
            
            # o mesmo que o yake, pouco mais custoso e resultado (teoricamente) melhor em extração de palavras-chave
            def theme_from_rake():
                try:
                    download_punkt() # se já existir, não faz nada
                    rake = Rake()
                    rake.extract_keywords_from_text(query)
                    print(f'len of keyword list = {len(rake.get_ranked_phrases())}')
                    return " ".join([ keyword for keyword in rake.get_ranked_phrases()])
                except Exception as e:
                    print("Error: ", str(e))
                    traceback.print_exc()
                    
            # especializado em sumarização extrema, baseado em transformer (pesado)
            def theme_from_pegasus():
                try:
                    model_name = "google/pegasus-xsum"
                    tokenizer = PegasusTokenizer.from_pretrained(model_name)
                    model = PegasusForConditionalGeneration.from_pretrained(model_name)
                    tokens = tokenizer(query, truncation=True, padding="longest", return_tensors="pt")
                    theme_tokens = model.generate(**tokens)
                    return tokenizer.decode(theme_tokens[0], skip_special_tokens=True)
                except Exception as e:
                    print("Error: ", str(e))
                    traceback.print_exc()
                    
            # BERT + GPT -> faz outras coisas, porém é bom pra sumarização. Alternativa mais pesada.
            def theme_from_bart():
                try:
                    model_name = "facebook/bart-large-cnn"
                    tokenizer = BartTokenizer.from_pretrained(model_name)
                    model = BartForConditionalGeneration.from_pretrained(model_name)
                    tokens = tokenizer(query, truncation=True, padding="longest", return_tensors="pt")
                    theme_tokens = model.generate(**tokens)
                    return tokenizer.decode(theme_tokens[0], skip_special_tokens=True)
                except Exception as e:
                    print("Error: ", str(e))
                    traceback.print_exc()
                    
            theme_of_query = query
            if option == 'yake':
                theme_of_query = theme_from_yake()
            elif option == 'rake':
                theme_of_query = theme_from_rake()
            elif option == 'pegasus':
                theme_of_query = theme_from_pegasus()
            elif option == 'bart':
                theme_of_query = theme_from_bart()
            
            return self.model.encode([theme_of_query])

        # olhar também aleatoriamente uma % muito pequena da nossa memória
        # Não melhora significativamente o resultado e fica grande demais pra ler e validar
        def get_rand_sample():
            if self.rand_sample_percent <= 0.0:
                return []
            total_items = len(self.df_data) // 2
            return [self.fetch_conversation(self.df_data, idx) for idx in np.random.choice(total_items, size=int(total_items*self.rand_sample_percent), replace=False)]
        
        try:
            if self.df_data is not None:
                # mantendo o padrão de busca em vizinhança do tema/query
                results = search_in_index(get_theme_of_query('bart')) 

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
