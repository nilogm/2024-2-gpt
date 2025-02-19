# Baseado em https://huggingface.co/blog/open-llm-leaderboard-mmlu

import torch
import random
import traceback
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from gpt_2024_2.model.brain_bot import BrainBot
from gpt_2024_2.eval.prompts import EXAM_PROMPT, LLM_EXAM_PROMPT, LLM_ANALYSIS_PROMPT
from gpt_2024_2.utils import load_env


hf_auth = load_env()


dict_choices = {0: "A", 1: "B", 2: "C", 3: "D"}
inverse_dict_choices = {"A": 0, "B": 1, "C": 2, "D": 3}


def format_choices(correct_answer, wrong_answer1, wrong_answer2, wrong_answer3):
    choices = [str(correct_answer), str(wrong_answer1), str(wrong_answer2), str(wrong_answer3)]

    random.shuffle(choices)

    correct_answer_index = choices.index(str(correct_answer))
    correct_answer = dict_choices[correct_answer_index]

    string_choices = ""
    for i in range(len(choices)):
        string_choices += f"- {dict_choices[i]}) {choices[i]}\n"

    return string_choices, correct_answer, choices


def calculate_multiple_choice_accuracy(rag: BrainBot, results: pd.DataFrame):
    df_results = results.copy()

    df_results["multiple_choice_correct_answer"] = ""
    df_results["multiple_choice_answer"] = ""
    df_results["multiple_choice_accuracy_score"] = float("nan")
    model_pipeline = rag.chat_module.model_pipeline

    # Get questions
    questions = df_results["question"]

    # Get contexts
    contexts = df_results["rag_context"]

    # Get correct answers
    correct_answers = df_results["correct_answer"]
    wrong_answers1 = df_results["wrong_answer1"]
    wrong_answers2 = df_results["wrong_answer2"]
    wrong_answers3 = df_results["wrong_answer3"]

    base_prompt = EXAM_PROMPT

    # Concat prompts
    prompts = pd.DataFrame(columns=["prompt", "correct_answer", "choices"])
    for i in range(len(questions)):
        context = contexts[i]
        question = questions[i]
        choices, correct_answer, choices_list = format_choices(correct_answers[i], wrong_answers1[i], wrong_answers2[i], wrong_answers3[i])
        prompt = base_prompt.format(context=context, question=question, choices=choices)
        prompts = pd.concat([prompts, pd.DataFrame({"prompt": [prompt], "correct_answer": [correct_answer], "choices": [choices_list]})], ignore_index=True)

    dataset_prompts = Dataset.from_pandas(prompts)
    try:
        count = 0
        pbar = tqdm(total=len(dataset_prompts))
        for out in model_pipeline(KeyDataset(dataset_prompts, "prompt"), batch_size=1, max_new_tokens=3):
            torch.cuda.empty_cache()
            for seq in out:
                line = prompts.iloc[count]
                prompt = line["prompt"]
                correct_answer = line["correct_answer"]
                choices = line["choices"]
                response = seq["generated_text"][len(prompt) :]

                answer = str(response).strip().upper()
                if answer != "":
                    answer = answer[0]

                df_results.at[count, "multiple_choice_correct_answer"] = correct_answer
                answer_str = str(answer)
                if answer in inverse_dict_choices:
                    answer_str += " - " + str(choices[inverse_dict_choices[answer]])
                df_results.at[count, "multiple_choice_answer"] = answer_str
                df_results.at[count, "multiple_choice_accuracy_score"] = 1 if answer == str(correct_answer).strip()[0] else 0

                pbar.set_description(f"Accuracy: {df_results[df_results['wrong_answer1'].notna()]['multiple_choice_accuracy_score'].mean():.2f}")
                count += 1
                inc = count - pbar.n
                pbar.update(n=inc)

        return df_results

    except Exception as e:
        traceback.print_exc()
        print("Error: ", str(e))
        print("Error in evaluating multiple choice questions!!")
        return results


def calculate_multiple_choice_accuracy_llm(model, results: pd.DataFrame):
    df_results = results.copy()

    df_results["multiple_choice_llm_correct_answer"] = ""
    df_results["multiple_choice_llm_answer"] = ""
    df_results["multiple_choice_llm_accuracy_score"] = float("nan")

    # Get correct answers
    correct_answers = df_results["correct_answer"]
    wrong_answers1 = df_results["wrong_answer1"]
    wrong_answers2 = df_results["wrong_answer2"]
    wrong_answers3 = df_results["wrong_answer3"]
    answers = df_results["system_answer"]

    base_prompt = LLM_EXAM_PROMPT

    # Concat prompts
    prompts = pd.DataFrame(columns=["prompt", "correct_answer", "choices"])
    for i in range(len(df_results)):
        choices, correct_answer, choices_list = format_choices(correct_answers[i], wrong_answers1[i], wrong_answers2[i], wrong_answers3[i])
        prompt = base_prompt.format(answer=answers[i], choices=choices)
        prompts = pd.concat([prompts, pd.DataFrame({"prompt": [prompt], "correct_answer": [correct_answer], "choices": [choices_list]})], ignore_index=True)

    dataset_prompts = Dataset.from_pandas(prompts)
    try:
        count = 0
        pbar = tqdm(total=len(dataset_prompts))
        for out in model(KeyDataset(dataset_prompts, "prompt"), batch_size=1, max_new_tokens=3):
            torch.cuda.empty_cache()
            for seq in out:
                line = prompts.iloc[count]
                prompt = line["prompt"]
                correct_answer = line["correct_answer"]
                choices = line["choices"]
                response = seq["generated_text"][len(prompt) :]

                answer = str(response).strip().upper()
                if answer != "":
                    answer = answer[0]

                df_results.at[count, "multiple_choice_llm_correct_answer"] = correct_answer
                answer_str = str(answer)
                if answer in inverse_dict_choices:
                    answer_str += " - " + str(choices[inverse_dict_choices[answer]])
                df_results.at[count, "multiple_choice_llm_answer"] = answer_str
                df_results.at[count, "multiple_choice_llm_accuracy_score"] = 1 if answer == str(correct_answer).strip()[0] else 0

                pbar.set_description(f"Accuracy: {df_results[df_results['wrong_answer1'].notna()]['multiple_choice_llm_accuracy_score'].mean():.2f}")
                count += 1
                inc = count - pbar.n
                pbar.update(n=inc)

        return df_results

    except Exception as e:
        traceback.print_exc()
        print("Error: ", str(e))
        print("Error in evaluating LLM multiple choice questions!")
        return results


def llm_analysis(model, results: pd.DataFrame):
    df_results = results.copy()

    df_results["big_llm_analysis"] = float("nan")

    # Get correct answers
    questions = df_results["question"]
    correct_answers = df_results["correct_answer"]
    wrong_answers1 = df_results["wrong_answer1"]
    wrong_answers2 = df_results["wrong_answer2"]
    wrong_answers3 = df_results["wrong_answer3"]
    answers = df_results["system_answer"]

    base_prompt = LLM_ANALYSIS_PROMPT

    prompts = pd.DataFrame(columns=["prompt"])
    for i in range(len(df_results)):
        prompt = base_prompt.format(question=questions[i], answer=answers[i], a1=correct_answers[i], a2=wrong_answers1[i], a3=wrong_answers2[i], a4=wrong_answers3[i])
        prompts = pd.concat([prompts, pd.DataFrame({"prompt": [prompt]})], ignore_index=True)

    dataset_prompts = Dataset.from_pandas(prompts)
    try:
        count = 0
        pbar = tqdm(total=len(dataset_prompts))
        pbar.set_description(f"Analysing answers...")

        for out in model(KeyDataset(dataset_prompts, "prompt"), batch_size=5):
            torch.cuda.empty_cache()
            for seq in out:
                line = prompts.iloc[count]
                prompt = line["prompt"]
                response = seq["generated_text"][len(prompt) :]
                df_results.at[count, "big_llm_analysis"] = response

                count += 1
                inc = count - pbar.n
                pbar.update(n=inc)

        return df_results

    except Exception as e:
        traceback.print_exc()
        print("Error: ", str(e))
        print("Error while getting LLM response analysis!")
        return results
