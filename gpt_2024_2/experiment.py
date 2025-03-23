import os
import traceback
import pandas as pd
from tqdm import tqdm
from gpt_2024_2.model.brain_bot import BrainBot


def test_qa(model: BrainBot, generative_codename: str, encoder_codename: str, results_dir: str, qa_file: str):
    df = pd.read_csv(qa_file, comment="#")

    filename = f"{generative_codename}__{encoder_codename}__{model.retriever.top_k}__{model.summarizer}.csv"
    csv_path = os.path.join(results_dir, filename)
    print(f"Generating answers and saving in {csv_path}")

    try:
        df["system_answer"] = ""
        df["dates"] = ""
        df["conversations"] = ""
        df["conversations_idx"] = ""
        df["relevance_check_time"] = ""
        df["semantic_conversations_idx"] = ""
        df["semantic_retrieval_time"] = ""
        df["date_conversations_idx"] = ""
        df["date_retrieval_time"] = ""

        pbar = tqdm(range(len(df.index)))
        for idx in pbar:
            pbar.set_description("Answering questions...")

            row = df.iloc[idx]
            question = row["question"]
            response, conversations, date, semantic_retrieval, date_retrieval = model.ask(question)

            df.at[idx, "system_answer"] = response
            df.at[idx, "dates"] = date[0]
            df.at[idx, "date_retrieval_time"] = date[1]
            df.at[idx, "conversations"] = conversations[0]
            df.at[idx, "conversations_idx"] = conversations[1]
            df.at[idx, "relevance_check_time"] = conversations[2]
            df.at[idx, "semantic_conversations_idx"] = [i[0] for i in semantic_retrieval[0]]
            df.at[idx, "semantic_retrieval_time"] = semantic_retrieval[1]
            df.at[idx, "date_conversations_idx"] = [i[0] for i in date_retrieval[0]]
            df.at[idx, "date_retrieval_time"] = date_retrieval[1]

            pbar.update(n=idx - pbar.n)

    except Exception as e:
        print("Error: ", str(e))
        traceback.print_exc()
        print("Error while generating answers: ", filename)
        print("Saving current progress...")

    print(f"Saving to {csv_path}")
    df.to_csv(csv_path, index=False)
