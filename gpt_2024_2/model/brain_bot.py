import os
import gc
import torch
from typing import Tuple
from dotenv import load_dotenv
import gpt_2024_2.context.builder as builder
from gpt_2024_2.model.gen_model import init_model
from gpt_2024_2.model.prompts import CHAT_PROMPT, build_message, DATE_PROMPT
from gpt_2024_2.model.retriever import SentenceRetriever
from gpt_2024_2.utils import path_check

load_dotenv()


class BrainBot:
    def __init__(self, device=0) -> None:
        self.params = {"model_id": None, "encoder_model_id": None}
        self.device = device

    def unload(self):
        self._terminate_model()
        self._terminate_retriever()

    def setup_models(self, generator: str = None, encoder: str = None, verbose: bool = False, **kwargs):
        self.hf_auth = os.getenv("HUGGINGFACE_AUTH_TOKEN")

        encoder_codename, encoder_model_id, self.encoder_model_dimensions = encoder
        self.data_dir = path_check(f"data/{encoder_codename}")

        _, model_id = generator

        self.setup_context()

        self.initialize_model(model_id)
        self.initialize_retriever(encoder_model_id, **kwargs)
        self.retriever.load()

        self.params = kwargs
        self.params.update({"model_id": model_id, "encoder_model_id": encoder_model_id})

        self.initialize_context(verbose=verbose)
        self.retriever.load_database(self.data)

    def setup_context(self):
        corpus = str(self.data_dir.joinpath(f"data.csv"))
        index = str(self.data_dir.joinpath(f"data.index"))
        self.data = {"corpus": corpus, "index": index, "destination": self.data_dir, "key": "data"}

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

    def initialize_model(self, model_id: str):
        if self.params["model_id"] == model_id:
            return

        self._terminate_model()
        self.model_pipeline, self.model, self.tokenizer = init_model(model_id, hf_auth=self.hf_auth, verbose=True, device_map=self.device)

    def _terminate_retriever(self):
        if hasattr(self, "retriever") and self.retriever:
            self.retriever.unload()

        gc.collect()
        torch.cuda.empty_cache()

    def initialize_retriever(self, encoder_model_id: str, **kwargs):
        if encoder_model_id == self.params["encoder_model_id"]:
            return

        self._terminate_retriever()
        print(f"Initializing {self.params['encoder_model_id']}")
        self.retriever = SentenceRetriever(encoder_model_id=encoder_model_id, **kwargs)

    def initialize_context(self, verbose: bool = True):
        index = dict(encoder_model=self.retriever.model, encoder_model_dimensions=self.encoder_model_dimensions, verbose=verbose)
        main_source = os.path.join("dataset", "memories")

        if not os.path.exists(self.data["corpus"]):
            builder.build_chunks(main_source, self.data["destination"], self.data["key"], verbose=verbose)

        if not os.path.exists(self.data["index"]):
            builder.build_index(self.data["corpus"], self.data["index"], **index)

    def _get_response(self, query: str) -> str:
        self.model_pipeline.call_count = 0
        sequences = self.model_pipeline(query)
        torch.cuda.empty_cache()
        response = sequences[0]["generated_text"]
        return response.strip()

    def format_message(self, message: str) -> Tuple[str, str, float]:
        dates = self.extract_dates(message)

        results, time_retrieval = self.retriever.retrieve_by_date(dates) if len(dates["dates"]) > 0 else self.retriever.retrieve(message, dates)

        conversations_idx = [i[0] for i in results]
        conversations = [i[1] for i in results]
        context = "\n\n".join(conversations)

        prompt = CHAT_PROMPT.format(conversations=context, today="Sunday, March 9th, 2025")
        formatted_message = build_message(self.tokenizer, message, prompt)

        return formatted_message, dates, context, conversations_idx, time_retrieval

    def extract_dates(self, message: str):
        prompt = DATE_PROMPT.format(today="Sunday, March 09th, 2025")
        formatted_message = build_message(self.tokenizer, message, prompt)
        dates = self._get_response(formatted_message)
        try:
            return eval(dates)
        except:
            return {"dates": []}

    def ask(self, message: str) -> Tuple[str, str, int, float, float]:
        query, dates, conversations, conversations_idx, time_retrieval = self.format_message(message)
        response = self._get_response(query)
        return response, dates, conversations, conversations_idx, time_retrieval
