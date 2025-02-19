from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import SemanticSimilarity, RougeScore
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
import pandas as pd


def to_ragas_dataset(results: pd.DataFrame, key: str = "correct_answer"):
    def to_sample(row):
        return SingleTurnSample(response=row["system_answer"], reference=row[key]).to_dict()

    dataset = results.apply(to_sample, axis=1)
    return EvaluationDataset.from_list(dataset.to_list())


def ragas_semantic(wrapped_llm: LangchainLLMWrapper, wrapped_embeddings: LangchainEmbeddingsWrapper, results: pd.DataFrame):
    results = results.copy()

    ragas_dataset = to_ragas_dataset(results)
    result = evaluate(dataset=ragas_dataset, llm=wrapped_llm, metrics=[SemanticSimilarity(embeddings=wrapped_embeddings), RougeScore(measure_type="recall")], raise_exceptions=True).to_pandas()
    results["semantic_similarity_score"] = result["semantic_similarity"]
    print(f"RAGAS's Semantic Similarity Score: ", result["semantic_similarity"].mean())
    results["rouge_score"] = result["rouge_score"]
    print(f"RAGAS's Rouge-L Recall Score: ", result["rouge_score"].mean())

    if "wrong_answer1" in results.columns and results["wrong_answer1"].notna().all():
        ragas_dataset = to_ragas_dataset(results, "wrong_answer1")
        result = evaluate(dataset=ragas_dataset, llm=wrapped_llm, metrics=[SemanticSimilarity(embeddings=wrapped_embeddings), RougeScore(measure_type="recall")], raise_exceptions=True).to_pandas()
        results["semantic_similarity_wa1_score"] = result["semantic_similarity"]
        print(f"RAGAS's Semantic Similarity Score: ", result["semantic_similarity"].mean())
        results["rouge_wa1_score"] = result["rouge_score"]
        print(f"RAGAS's Rouge-L Recall Score: ", result["rouge_score"].mean())

    if "wrong_answer2" in results.columns and results["wrong_answer2"].notna().all():
        ragas_dataset = to_ragas_dataset(results, "wrong_answer2")
        result = evaluate(dataset=ragas_dataset, llm=wrapped_llm, metrics=[SemanticSimilarity(embeddings=wrapped_embeddings), RougeScore(measure_type="recall")], raise_exceptions=True).to_pandas()
        results["semantic_similarity_wa2_score"] = result["semantic_similarity"]
        print(f"RAGAS's Semantic Similarity Score: ", result["semantic_similarity"].mean())
        results["rouge_wa2_score"] = result["rouge_score"]
        print(f"RAGAS's Rouge-L Recall Score: ", result["rouge_score"].mean())

    if "wrong_answer3" in results.columns and results["wrong_answer3"].notna().all():
        ragas_dataset = to_ragas_dataset(results, "wrong_answer3")
        result = evaluate(dataset=ragas_dataset, llm=wrapped_llm, metrics=[SemanticSimilarity(embeddings=wrapped_embeddings), RougeScore(measure_type="recall")], raise_exceptions=True).to_pandas()
        results["semantic_similarity_wa3_score"] = result["semantic_similarity"]
        print(f"RAGAS's Semantic Similarity Score: ", result["semantic_similarity"].mean())
        results["rouge_wa3_score"] = result["rouge_score"]
        print(f"RAGAS's Rouge-L Recall Score: ", result["rouge_score"].mean())

    return results
