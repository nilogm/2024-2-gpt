import pandas as pd
import os
import traceback
import torch

import gpt_2024_2.model.gen_model as gen_model
from gpt_2024_2.eval.classical_metrics import calculate_bert_score
from gpt_2024_2.eval.ragas_semantic import ragas_semantic
from gpt_2024_2.utils import load_env

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper


hf_auth = load_env()

metrics = [
    "semantic_similarity_score",
    "rouge_score_recall",
    "rouge_score_precision",
    "rouge_score_f1",
    "bert_recall_score",
    "bert_precision_score",
]


def evaluate(results_dir: str, out_dir: str, device: int):

    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs=dict(device=device, trust_remote_code=True))
    wrapped_embeddings = LangchainEmbeddingsWrapper(embed_model)
    model, _, _ = gen_model.init_model(model_id="meta-llama/Llama-3.2-1B-Instruct", hf_auth=hf_auth, quantization=4, device_map=device)
    wrapped_llm = LangchainLLMWrapper(model)

    # Run evaluation for every file in results_dir
    list_files = os.listdir(results_dir)
    list_files.sort()
    final_scores = pd.DataFrame()

    for filename in list_files:
        if filename.endswith(".csv"):
            try:
                csv_path = os.path.join(results_dir, filename)
                df = pd.read_csv(csv_path)
                print("\nEvaluating ", csv_path)

                out = ragas_semantic(wrapped_llm, wrapped_embeddings, df)
                out = calculate_bert_score(df, device=device)
                out.to_csv(os.path.join(out_dir, f"{filename.removesuffix('.csv')}__scores.csv"))
                final_scores = pd.concat([final_scores, out])

                torch.cuda.empty_cache()

            except Exception as e:
                traceback.print_exc()
                print("Error: ", str(e))
                print("Error while evaluating: ", csv_path)
