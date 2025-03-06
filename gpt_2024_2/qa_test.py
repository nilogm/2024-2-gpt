import os
import traceback
import pandas as pd
from tqdm import tqdm
from gpt_2024_2.model.brain_bot import BrainBot


def test_qa(model: BrainBot, generative_codename: str, encoder_codename: str, results_dir: str, test_dir: str, qa_file: str):
    csv_path = os.path.join(test_dir, qa_file)
    df = pd.read_csv(csv_path)

    df.reset_index(inplace=True)
    df["qa_file"] = qa_file

    filename = f"{qa_file.removesuffix('.csv').split('_')[0]}__{generative_codename}__{encoder_codename}.csv"

    csv_path = os.path.join(results_dir, filename)

    print(f"Generating answers and saving in {filename}")

    try:
        df["system_answer"] = ""
        df["rag_context"] = ""
        df["llm_response"] = ""
        df.drop(columns=["llm_response"], inplace=True)

        pbar = tqdm(total=len(df.index))
        count = 0
        for idx in range(len(df.index)):
            pbar.set_description("Answering questions...")

            row = df.iloc[idx]
            question = row["question"]
            response, rag_context, memory_used, retrieval_time, query_time = model.ask(question)

            df.at[idx, "system_answer"] = response
            df.at[idx, "rag_context"] = rag_context
            df.at[idx, "memory_used"] = memory_used
            df.at[idx, "retrieval_time"] = retrieval_time
            df.at[idx, "query_time"] = query_time

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
