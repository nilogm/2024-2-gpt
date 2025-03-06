import os
import gc
import time
import torch
from typing import Tuple
from dotenv import load_dotenv
import gpt_2024_2.context.builder as builder
from gpt_2024_2.model.gen_model import init_model
from gpt_2024_2.model.prompts import CHAT_PROMPT, build_message
from gpt_2024_2.model.retriever import SentenceRetriever
from gpt_2024_2.utils import get_encoder, get_model

load_dotenv()


class BrainBot:
    def __init__(self, generative_codename: str, encoder_codename: str, top_k: int = 3, borders: int = 1, device=0, init: bool = False, **kwargs) -> None:
        self.generative_codename = generative_codename
        self.encoder_codename = encoder_codename
        self.device = device
        self.init = init
        self.setup_models(generative_codename, encoder_codename, top_k=top_k, borders=borders, **kwargs)

    def load(self):
        if self.params["model_id"] is not None:
            self.initialize_model(**self.params)
        self.retriever.load()
        if not self.init:
            self.load_memory()

    def unload(self):
        self._terminate_model()
        self._terminate_retriever()

    def setup_models(self, generative_codename: str = None, encoder_codename: str = None, top_k: int = 3, borders: int = 1, **kwargs):
        self.hf_auth = os.getenv("HUGGINGFACE_AUTH_TOKEN")

        encoder_model_id, self.encoder_model_dimensions, self.data_dir = get_encoder(encoder_codename)
        if encoder_model_id is None:
            print("Encoder is set to 'None'. Not using RAG system.")
        self.use_rag = encoder_model_id is not None

        model_id = get_model(generative_codename)
        if model_id is None and not self.init:
            raise ValueError("Generative model cannot be 'None'.")

        self.params = kwargs
        self.params.update(dict(model_id=model_id, encoder_model_id=encoder_model_id, top_k=top_k, init=self.init))

        self.base_prompt = CHAT_PROMPT

        self.initialize_retriever(**self.params)
        self.setup_files()

    def setup_files(self):
        corpus = str(self.data_dir.joinpath(f"data.csv"))
        index = str(self.data_dir.joinpath(f"data.index"))
        self.data = {"corpus": corpus, "index": index, "destination": self.data_dir, "key": "data"}

    def load_memory(self):
        self.retriever.load_database(self.data)

    def _terminate_model(self):
        if hasattr(self, "model_pipeline") and self.model_pipeline:
            del self.model_pipeline
        if hasattr(self, "model") and self.model:
            del self.model
        if hasattr(self, "tokenizer") and self.tokenizer:
            del self.tokenizer

        self.model_pipeline = None
        self.model = None
        self.tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()

    def initialize_model(self, model_id: str, init: bool = False, **kwargs):
        self._terminate_model()

        print(f"Initializing {model_id}...")
        self.model_id = model_id
        self.model_pipeline, self.model, self.tokenizer = init_model(model_id, hf_auth=self.hf_auth, verbose=True, init=init, device_map=self.device)

    def _terminate_retriever(self):
        if hasattr(self, "retriever") and self.retriever:
            self.retriever.unload()

        gc.collect()
        torch.cuda.empty_cache()

    def initialize_retriever(self, encoder_model_id: str, top_k: int = 3, borders: int = 1, **kwargs):
        if hasattr(self, "encoder_model_id") and self.encoder_model_id == encoder_model_id:
            return

        self._terminate_retriever()

        if encoder_model_id is None:
            self.encoder_model_id = None
            self.retriever = None
            return

        print(f"Initializing {encoder_model_id}")
        self.encoder_model_id = encoder_model_id
        self.retriever = SentenceRetriever(encoder_model_id, top_k=top_k, borders=borders, device=self.device)

    def initialize_context(
        self,
        verbose: bool = True,
        **kwargs,
    ):
        index = dict(encoder_model=self.retriever.model, encoder_model_dimensions=self.encoder_model_dimensions, verbose=verbose)

        main_source = os.path.join("conversations", "dataset")
        builder.build_chunks(main_source, self.data["destination"], self.data["key"], verbose=verbose)
        builder.build_index(self.data["corpus"], self.data["index"], **index)

    def _get_response(self, query: str):
        self.model_pipeline.call_count = 0
        t = time.time()

        sequences = self.model_pipeline(query)

        memory_used = 0
        for i in os.popen("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits").read().split("\n"):
            pid, mem = i.split(",")
            pid = int(pid.strip())
            if pid == os.getpid():
                memory_used = int(mem.strip())
                break

        torch.cuda.empty_cache()
        et = time.time()

        response = sequences[0]["generated_text"]
        return response.strip(), memory_used, et - t

    def format_message(self, message: str, history: list, memory_limit: int = 0) -> Tuple[str, str, float]:
        results_books, time_retrieval = self.retriever.retrieve(message)
        text_books = "\n\n".join(results_books)

        if memory_limit == 0:
            history = []
        elif len(history) > memory_limit:
            history = history[-memory_limit:]

        prompt = self.base_prompt.format(text_books=text_books)
        formatted_message = build_message(self.tokenizer, message, history, prompt)

        return formatted_message, text_books, time_retrieval

    def ask(self, message: str, history: list[dict] = [], memory_limit: int = 0) -> Tuple[str, str, int, float, float]:
        print(f"Valor de 'message': {message} (Tipo: {type(message)})")
        query, rag_context, time_retrieval = self.format_message(message, history, memory_limit)
        response, memory_used, time_response = self._get_response(query)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

        return response, rag_context, memory_used, time_retrieval, time_response
