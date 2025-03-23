import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import torch
import traceback
import pandas as pd
from evaluate import load

from ragas import evaluate as ragas_evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import SemanticSimilarity, RougeScore
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

import gpt_2024_2.model.gen_model as gen_model
from gpt_2024_2.utils import load_env


hf_auth = load_env()


def evaluate(results_dir: str, out_dir: str, device: int):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs=dict(device=device, trust_remote_code=True))
    wrapped_embeddings = LangchainEmbeddingsWrapper(embed_model)
    model, _, _ = gen_model.init_model(model_id="meta-llama/Llama-3.2-1B-Instruct", hf_auth=hf_auth, device_map=device)
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
                out = calculate_bert_score(out, device=device)
                out = retrieval_scores(out)
                out.to_csv(os.path.join(out_dir, f"{filename.removesuffix('.csv')}__scores.csv"))
                final_scores = pd.concat([final_scores, out])

                torch.cuda.empty_cache()

            except Exception as e:
                traceback.print_exc()
                print("Error: ", str(e))
                print("Error while evaluating: ", csv_path)


def calculate_bert_score(results: pd.DataFrame, device: int = 0):
    print("Calculating BERT scores...")

    def evaluate_bert_score(row):
        bertscore = load("bertscore")
        predictions = row[["system_answer"]]
        references = row[["answer"]]

        row["bert_precision_score"] = float("nan")
        try:
            results = bertscore.compute(predictions=predictions, references=references, device=device, model_type="roberta-large", idf=True)
            torch.cuda.empty_cache()
            row["bert_precision_score"] = results["precision"][0] * 100
        except Exception as e:
            print(traceback.print_exc(e))

        return row

    dataset = results.copy()
    dataset = dataset.apply(evaluate_bert_score, axis=1)
    print("BERT Precision: ", dataset["bert_precision_score"].mean())

    return dataset


def ragas_semantic(wrapped_llm: LangchainLLMWrapper, wrapped_embeddings: LangchainEmbeddingsWrapper, results: pd.DataFrame):
    results = results.copy()

    def to_sample(row):
        return SingleTurnSample(response=row["system_answer"], reference=row["answer"]).to_dict()

    dataset = results.apply(to_sample, axis=1)
    ragas_dataset = EvaluationDataset.from_list(dataset.to_list())

    result = ragas_evaluate(dataset=ragas_dataset, llm=wrapped_llm, metrics=[SemanticSimilarity(embeddings=wrapped_embeddings), RougeScore(measure_type="recall")], raise_exceptions=True).to_pandas()
    results["semantic_similarity_score"] = result["semantic_similarity"] * 100
    print(f"RAGAS's Semantic Similarity Score: ", results["semantic_similarity_score"].mean())
    results["rouge_score_recall"] = result["rouge_score"] * 100
    print(f"RAGAS's Rouge-L Recall Score: ", results["rouge_score_recall"].mean())

    result = ragas_evaluate(dataset=ragas_dataset, llm=wrapped_llm, metrics=[RougeScore(measure_type="precision")], raise_exceptions=True).to_pandas()
    results["rouge_score_precision"] = result["rouge_score"] * 100
    print(f"RAGAS's Rouge-L Precision Score: ", results["rouge_score_precision"].mean())

    result = ragas_evaluate(dataset=ragas_dataset, llm=wrapped_llm, metrics=[RougeScore(measure_type="fmeasure")], raise_exceptions=True).to_pandas()
    results["rouge_score_f1"] = result["rouge_score"] * 100
    print(f"RAGAS's Rouge-L F1 Score: ", results["rouge_score_f1"].mean())

    return results


def retrieval_scores(df: pd.DataFrame):
    df["date_retrieved"] = df.apply(lambda x: int(x["conversation_id"] in eval(x["date_conversations_idx"])) * 100, axis=1)
    df["date_retrieved_right_date"] = df.apply(lambda x: (int(x["date_retrieved"])) if x["right_date"] else float("nan"), axis=1)
    df["semantic_retrieved"] = df.apply(lambda x: int(x["conversation_id"] in eval(x["semantic_conversations_idx"])) * 100, axis=1)
    df["semantic_or_date_retrieved"] = df.apply(lambda x: min(100, x["semantic_retrieved"] + x["date_retrieved"]), axis=1)
    df["relevant_retrieved"] = df.apply(lambda x: int(x["conversation_id"] in eval(x["conversations_idx"])) * 100, axis=1)
    df["relevant_retrieved_specific"] = df.apply(lambda x: int(x["relevant_retrieved"] and len(eval(x["conversations_idx"])) == 1) * 100, axis=1)

    print(f"Date retrieved: ", df["date_retrieved"].mean())
    print(f"Date retrieved (when needed): ", df["date_retrieved_right_date"].mean())
    print(f"Semantic retrieved: ", df["semantic_retrieved"].mean())
    print(f"Date or Semantic retrieved: ", df["semantic_or_date_retrieved"].mean())
    print(f"Relevant retrieved: ", df["relevant_retrieved"].mean())
    print(f"Specific Relevant retrieved: ", df["relevant_retrieved_specific"].mean())

    return df
