import os
import gc
import re
import time
import torch
import datetime
import numpy as np
from typing import Tuple
from dotenv import load_dotenv
from gpt_2024_2.model.builder import build_chunks, build_index
from gpt_2024_2.model.gen_model import init_model
from gpt_2024_2.model.prompts import CHAT_PROMPT, DATE_PROMPT, build_message, RELEVANCY_PROMPT
from gpt_2024_2.model.retriever import SentenceRetriever
from gpt_2024_2.utils import path_check

load_dotenv()


class BrainBot:
    def __init__(self, memories_path: str, memory_end_date: datetime.time, today_date: str, device=0) -> None:
        self.hf_auth = os.getenv("HUGGINGFACE_AUTH_TOKEN")
        self.memories_path = memories_path
        self.memory_end_date = memory_end_date
        self.today_date = today_date
        self.device = device

    def unload(self):
        self._terminate_model()
        self._terminate_retriever()

    def setup_models(self, generator: dict = None, encoder: dict = None, summarizer: str = "default", top_k: int = 3, verbose: bool = False):
        self.summarizer = summarizer

        self.encoder_model_dimensions = encoder["dimensions"]
        self.data_dir = path_check(f"data/{encoder['nickname']}_{summarizer}")
        model_id = generator["model_id"]

        self.initialize_model(model_id, verbose=verbose)
        self.initialize_retriever(encoder["model_id"], top_k)
        self.retriever.top_k = top_k
        self.retriever.load()

        self.initialize_context(verbose=verbose)

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

    def initialize_model(self, model_id: str, verbose: bool = False):
        if hasattr(self, "model_id") and self.model_id == model_id:
            return

        self._terminate_model()
        self.model_id = model_id
        self.model_pipeline, self.model, self.tokenizer = init_model(model_id, hf_auth=self.hf_auth, verbose=verbose, device_map=self.device)

    def _terminate_retriever(self):
        if hasattr(self, "retriever") and self.retriever:
            self.retriever.unload()

        gc.collect()
        torch.cuda.empty_cache()

    def initialize_retriever(self, encoder_model_id: str, top_k: int, verbose: bool = False):
        if hasattr(self, "encoder_model_id") and encoder_model_id == self.encoder_model_id:
            return

        self._terminate_retriever()
        if verbose:
            print(f"Initializing {encoder_model_id}")
        self.encoder_model_id = encoder_model_id
        self.retriever = SentenceRetriever(encoder_model_id=encoder_model_id, top_k=top_k, device=self.device)

    def initialize_context(self, verbose: bool = True):
        corpus = str(self.data_dir.joinpath("memories.csv"))
        index = str(self.data_dir.joinpath("memories.index"))

        if not os.path.exists(corpus):
            build_chunks(self.memories_path, self.data_dir, end_date=self.memory_end_date, verbose=verbose)
        if not os.path.exists(index):
            build_index(corpus, index, encoder_model=self.retriever.model, encoder_model_dimensions=self.encoder_model_dimensions, verbose=verbose)

        self.retriever.load_database(corpus, index)

    def _get_response(self, query: str) -> str:
        self.model_pipeline.call_count = 0
        sequences = self.model_pipeline(query)
        torch.cuda.empty_cache()
        response = sequences[0]["generated_text"]
        return response.strip()

    def format_message(self, message: str) -> Tuple[str, str, float]:
        dates, date_time_retrieval = self.extract_dates(message)

        date_results, date_memory_time_retrieval = [], 0
        if len(dates["dates"]) > 0:
            date_results, date_memory_time_retrieval = self.retriever.retrieve_by_date(dates)
        semantic_results, semantic_memory_time_retrieval = self.retriever.retrieve(message, dates)

        conversations_idx = [i[0] for i in date_results + semantic_results]
        conversations = [i[1] for i in date_results + semantic_results]

        relevant_items_idx, relevance_time = self.check_relevancy(message, conversations)
        context = "\n\n".join([conversations[i] for i in relevant_items_idx])

        prompt = CHAT_PROMPT.format(conversations=context, today=self.today_date)
        formatted_message = build_message(self.tokenizer, message, prompt)

        return (
            formatted_message,
            (context, [conversations_idx[i] for i in relevant_items_idx], relevance_time),
            (dates, date_time_retrieval),
            (semantic_results, semantic_memory_time_retrieval),
            (date_results, date_memory_time_retrieval),
        )

    def extract_dates(self, message: str):
        t = time.time()
        prompt = DATE_PROMPT.format(today=self.today_date)
        formatted_message = build_message(self.tokenizer, message, prompt)
        dates = self._get_response(formatted_message)
        et = time.time() - t

        try:
            dates_ = "{" + re.findall(r"{(.*)}", dates)[0] + "}"
            item = eval(dates_)
            if "dates" in item:
                return item, et
        except:
            pass

        return {"dates": []}, et

    def check_relevancy(self, message: str, conversations: list[str]):
        t = time.time()
        c = 0
        max_amount = 10
        relevant_items = []
        while c < len(conversations):
            convos = conversations[c : min(c + max_amount, len(conversations))]

            prompt = RELEVANCY_PROMPT.format(today=self.today_date, conversations="\n\n".join([f"Conversation {i}:\n{convo}" for i, convo in enumerate(convos, start=1)]))
            formatted_message = build_message(self.tokenizer, message, prompt)
            relevant_conversations = self._get_response(formatted_message)

            try:
                idx_json = "{" + re.findall(r"{(.*?)}", relevant_conversations)[0] + "}"
                items = [i - 1 + c for i in eval(idx_json)["relevant_conversations"]]
                relevant_items.extend(items)
            except:
                pass

            c += 10

        et = time.time() - t
        return np.unique(relevant_items), et

    def ask(self, message: str) -> Tuple[str, str, int, float, float]:
        query, conversations, date, semantic_results, date_results = self.format_message(message)
        response = self._get_response(query)
        return response, conversations, date, semantic_results, date_results
