import os
import traceback
import pandas as pd
from tqdm import tqdm
from gpt_2024_2.model.brain_bot import BrainBot


def test_qa(model: BrainBot, generative_codename: str, encoder_codename: str, results_dir: str, qa_file: str):
    df = pd.read_csv(qa_file, comment="#")

    filename = f"{generative_codename}__{encoder_codename}__{model.retriever.top_k}__{model.retriever.summarizer}.csv"
    csv_path = os.path.join(results_dir, filename)
    print(f"Generating answers and saving in {csv_path}")

    try:
        df["system_answer"] = ""
        df["dates"] = ""
        df["conversations"] = ""
        df["conversations_idx"] = ""
        df["retrieval_time"] = ""

        pbar = tqdm(total=len(df.index))
        count = 0
        for idx in range(len(df.index)):
            pbar.set_description("Answering questions...")

            row = df.iloc[idx]
            question = row["question"]
            response, dates, conversations, conversations_idx, retrieval_time = model.ask(question)

            df.at[idx, "system_answer"] = response
            df.at[idx, "dates"] = dates
            df.at[idx, "conversations"] = conversations
            df.at[idx, "conversations_idx"] = conversations_idx
            df.at[idx, "retrieval_time"] = retrieval_time

            count += 1
            inc = count - pbar.n
            pbar.update(n=inc)

    except Exception as e:
        print("Error: ", str(e))
        traceback.print_exc()
        print("Error while generating answers: ", filename)
        print("Saving current progress...")

    print(f"Saving to {csv_path}")
    df.to_csv(csv_path, index=False)
