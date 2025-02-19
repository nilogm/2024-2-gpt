import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import torch
import pandas as pd
import numpy as np
from evaluate import load

params = dict(model_type="bert-base-multilingual-cased", lang="pt", idf=True)


def calculate_bert_score(results: pd.DataFrame, device: int = 0):
    print("Calculating BERT scores...")

    def evaluate_bert_score(row):
        bertscore = load("bertscore")
        predictions = row[["system_answer", "system_answer", "system_answer", "system_answer"]]
        references = row[["correct_answer", "wrong_answer1", "wrong_answer2", "wrong_answer3"]]

        try:
            results = bertscore.compute(predictions=predictions, references=references, device=device, **params)
            torch.cuda.empty_cache()

            row["bert_recall_score"] = results["recall"][0]
            row["bert_precision_score"] = results["precision"][0]
            row["bert_f1_score"] = results["f1"][0]

            for i in range(1, 4):
                row[f"bert_recall_w{i}_score"] = results["recall"][i]
                row[f"bert_precision_w{i}_score"] = results["precision"][i]
                row[f"bert_f1_w{i}_score"] = results["f1"][i]

            row["bert_recall_choice_score"] = int(np.argmax(results["recall"][:-1]) == 0)
            row["bert_precision_choice_score"] = int(np.argmax(results["precision"][:-1]) == 0)
            row["bert_f1_choice_score"] = int(np.argmax(results["f1"][:-1]) == 0)

            if "complete_answer" in row:
                results = bertscore.compute(predictions=row[["system_answer"]], references=row[["complete_answer"]], device=device, **params)
                torch.cuda.empty_cache()

                row["bert_recall_correct_score"] = results["recall"][0]
                row["bert_precision_correct_score"] = results["precision"][0]
                row["bert_f1_correct_score"] = results["f1"][0]

        except:
            row["bert_recall_choice_score"] = float("nan")
            row["bert_precision_choice_score"] = float("nan")
            row["bert_f1_choice_score"] = float("nan")

            row["bert_recall_score"] = float("nan")
            row["bert_precision_score"] = float("nan")
            row["bert_f1_score"] = float("nan")

            for i in range(1, 4):
                row[f"bert_recall_w{i}_score"] = float("nan")
                row[f"bert_precision_w{i}_score"] = float("nan")
                row[f"bert_f1_w{i}_score"] = float("nan")

            if "complete_answer" in row:
                row["bert_recall_correct_score"] = float("nan")
                row["bert_precision_correct_score"] = float("nan")
                row["bert_f1_correct_score"] = float("nan")

        return row

    dataset = results.copy()
    dataset = dataset.apply(evaluate_bert_score, axis=1)
    print(
        "BERT Choice Recall/Precision/F1 Score: ",
        dataset[dataset["wrong_answer1"].notna()]["bert_recall_choice_score"].mean(),
        dataset[dataset["wrong_answer1"].notna()]["bert_precision_choice_score"].mean(),
        dataset[dataset["wrong_answer1"].notna()]["bert_f1_choice_score"].mean(),
    )

    return dataset
