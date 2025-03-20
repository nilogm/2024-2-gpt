import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import torch
import pandas as pd
import traceback
from evaluate import load

params = dict(model_type="bert-base-multilingual-cased", idf=True)


def calculate_bert_score(results: pd.DataFrame, device: int = 0):
    print("Calculating BERT scores...")

    def evaluate_bert_score(row):
        bertscore = load("bertscore")
        predictions = row[["system_answer"]]
        references = row[["correct_answer"]]
        
        row["bert_recall_score"] = float("nan")
        row["bert_precision_score"] = float("nan")
        row["bert_f1_score"] = float("nan")

        try:
            results = bertscore.compute(predictions=predictions, references=references, device=device, **params)
            torch.cuda.empty_cache()
            row["bert_recall_score"] = results["recall"][0]
            row["bert_precision_score"] = results["precision"][0]
            row["bert_f1_score"] = results["f1"][0]
        except Exception as e:
            print(traceback.print_exc(e))
            
        return row

    dataset = results.copy()
    dataset = dataset.apply(evaluate_bert_score, axis=1)
    print(
        "BERT Choice Recall/Precision/F1 Score: ",
        dataset[dataset["answer"].notna()]["bert_recall_score"].mean(),
        dataset[dataset["answer"].notna()]["bert_precision_score"].mean(),
        dataset[dataset["answer"].notna()]["bert_f1_score"].mean(),
    )

    return dataset
