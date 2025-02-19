import pandas as pd
import os
import traceback
import torch

import gpt_2024_2.eval.perplexity as perplexity
import gpt_2024_2.eval.multiple_choice_eval as multiple_choice_eval
import gpt_2024_2.model.gen_model as gen_model
from gpt_2024_2.eval.classical_metrics import calculate_bert_score
from gpt_2024_2.eval.ragas_semantic import ragas_semantic
from gpt_2024_2.model.brain_bot import BrainBot
from gpt_2024_2.utils import load_env

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper


hf_auth = load_env()

metrics = [
    "semantic_similarity_score",
    "semantic_similarity_short_score",
    "rouge_score",
    "rouge_short_score",
    "perplexity_score",
    "perplexity_accuracy_score",
    "multiple_choice_accuracy_score",
    "multiple_choice_llm_accuracy_score",
    "bert_recall_choice_score",
    "bert_precision_choice_score",
    "bert_recall_score",
    "bert_precision_score",
]
metadata = ["qa_file", "llm", "encoder", "questions", "summary", "vocabulary"]


def init_rag(rag: BrainBot, device: int, results: pd.DataFrame):
    params = dict(generative_codename=results["llm"].iloc[0], encoder_codename=results["encoder"].iloc[0])
    if rag is None:
        return BrainBot(device=device, **params)
    else:
        rag.set_configs(**params)
    return rag


def perplexity_eval(rag: BrainBot, results: pd.DataFrame, device: int = 0):
    try:
        out_ = perplexity.calculate_perplexity(rag, results, device=device)
        out_ = perplexity.calculate_perplexity_accuracy(rag, out_, device=device)
        return out_
    except Exception:
        traceback.print_exc()
        print("Could not calculate perplexity!")
        return results


def exam_eval(rag: BrainBot, results: pd.DataFrame):
    try:
        out_ = multiple_choice_eval.calculate_multiple_choice_accuracy(rag, results)
        return out_
    except Exception:
        traceback.print_exc()
        print("Could not calculate multiple choice accuracy!")
        return results


def llm_exam_eval(big_model, results: pd.DataFrame):
    try:
        out_ = multiple_choice_eval.calculate_multiple_choice_accuracy_llm(big_model, results)
        out_ = multiple_choice_eval.llm_analysis(big_model, results)
        return out_
    except Exception:
        traceback.print_exc()
        print("Could not calculate LLM multiple choice accuracy!")
        return results


def evaluate(results_dir: str, device: int, use_big_model: bool = False, outfile: str = "scores.csv"):
    rag: BrainBot = None

    wrapped_embeddings = None
    wrapped_llm = None
    if not use_big_model:
        embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs=dict(device=device, trust_remote_code=True))
        wrapped_embeddings = LangchainEmbeddingsWrapper(embed_model)
        model, _, _ = gen_model.init_model(model_id="meta-llama/Llama-3.2-1B-Instruct", hf_auth=hf_auth, quantization=4, device_map=device)
        wrapped_llm = LangchainLLMWrapper(model)

    big_model = None
    if use_big_model:
        big_model, _, _ = gen_model.init_model(model_id="meta-llama/Llama-3.1-70B-Instruct", hf_auth=hf_auth, quantization=8, device_map=device)

    # Run evaluation for every file in results_dir
    list_files = os.listdir(results_dir)
    list_files.sort()
    for filename in list_files:
        if filename.endswith(".csv") and filename != outfile:
            try:
                # Load the dataset with questions and answers
                csv_path = os.path.join(results_dir, filename)
                df = pd.read_csv(csv_path)
                print("\nEvaluating ", csv_path)

                out = None
                for llm in df["llm"].unique():
                    for encoder in df["encoder"].unique():
                        if len(df[(df["llm"] == llm) & (df["encoder"] == encoder)]) == 0:
                            continue

                        if use_big_model:
                            out_ = llm_exam_eval(big_model, df[(df["llm"] == llm) & (df["encoder"] == encoder)].reset_index())
                        else:
                            out_ = df[(df["llm"] == llm) & (df["encoder"] == encoder)].reset_index()
                            rag = init_rag(rag, device, out_)
                            out_ = perplexity_eval(rag, out_, device=device)
                            out_ = exam_eval(rag, out_)
                            out_ = ragas_semantic(wrapped_llm, wrapped_embeddings, out_)
                            out_ = calculate_bert_score(out_, device=device)

                        out = pd.concat([out, out_], ignore_index=True) if out is not None else out_
                        torch.cuda.empty_cache()

                if out is not None:
                    out.to_csv(os.path.join(results_dir, f"{filename.removesuffix('.csv')}_metrics.csv"))

            except Exception as e:
                traceback.print_exc()
                print("Error: ", str(e))
                print("Error while evaluating: ", csv_path)
