# https://huggingface.co/docs/transformers/v4.15.0/perplexity

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from gpt_2024_2.model.brain_bot import BrainBot
from gpt_2024_2.model.prompts import build_message
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from torch.nn import CrossEntropyLoss


def setup_prompts(prompt: str, tokenizer, df_results: pd.DataFrame):
    questions = df_results["question"]
    contexts = df_results["rag_context"]
    answers = df_results["correct_answer"]

    prompts_answers = [build_message(tokenizer, question, [], prompt.format(text_books=context), a) for context, question, a in zip(contexts, questions, answers)]
    prompts = [build_message(tokenizer, question, [], prompt.format(text_books=context)) for context, question in zip(contexts, questions)]
    return prompts_answers, prompts


def perplexity(model, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, prompt: str, prompt_answer: str, device: int = 0):
    prompt_len = len(tokenizer(prompt, return_tensors="pt").to(f"cuda:{device}").input_ids[0])
    encodings = tokenizer(prompt_answer, return_tensors="pt")
    input_ids = encodings.input_ids.to(f"cuda:{device}")[:, :]
    target_ids = input_ids.clone()
    target_ids[:, :prompt_len] = -100  # Coloca o contexto em -100 para que não afete na pontuação.

    with torch.no_grad():
        loss = model.forward(input_ids=input_ids, labels=target_ids).loss
        ppl = float(torch.exp(loss))

    return float(ppl)


def new_perplexity(model, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, prompt, prompt_answer, device: int = 0):
    encodings = tokenizer(
        prompt_answer,
        add_special_tokens=False,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(f"cuda:{device}")

    prompt_encodings = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(f"cuda:{device}")

    prompt_len = len(prompt_encodings["input_ids"][0])

    encoded_text = encodings["input_ids"]
    attn_mask = encodings["attention_mask"]

    loss_fct = CrossEntropyLoss(reduction="none")

    labels = encoded_text
    with torch.no_grad():
        out_logits = model(encoded_text, attention_mask=attn_mask).logits

    shift_logits = out_logits[..., prompt_len:-1, :].contiguous()
    shift_labels = labels[..., prompt_len + 1 :].contiguous()
    shift_attention_mask_batch = attn_mask[..., prompt_len + 1 :].contiguous()

    perplexity_batch = torch.exp((loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1) / shift_attention_mask_batch.sum(1))
    return perplexity_batch.tolist()[0]


def calculate_perplexity(rag: BrainBot, results: pd.DataFrame, device: int = 0):
    df_results = results.copy()

    df_results["perplexity_score"] = float(0)

    model, tokenizer = rag.model, rag.tokenizer
    prompts_answers, prompts = setup_prompts(rag.base_prompt, tokenizer, df_results)

    pbar = tqdm(total=len(df_results))
    pbar.set_description("Calculating perplexity...")
    count = 0
    for i in range(len(prompts)):
        try:
            df_results.at[i, "perplexity_score"] = perplexity(model, tokenizer, prompts[i], prompts_answers[i], device=device)
        except:
            df_results.at[i, "perplexity_score"] = 0

        torch.cuda.empty_cache()

        pbar.set_description(f"Calculating perplexity... (mean={df_results[df_results['wrong_answer1'].notna()]['perplexity_score'].mean():.2f})")

        count += 1
        inc = count - pbar.n
        pbar.update(n=inc)

    return df_results


def calculate_perplexity_accuracy(rag: BrainBot, results: pd.DataFrame, device: int = 0):
    df_results = results.copy()

    df_results["perplexity_accuracy_score"] = int(0)
    df_results["perplexity_accuracy_choice"] = int(0)

    df_results["perplexity_w1_score"] = float("nan")
    df_results["perplexity_w2_score"] = float("nan")
    df_results["perplexity_w3_score"] = float("nan")

    # Get correct answers
    correct_answers = df_results["correct_answer"]
    wrong_answers1 = df_results["wrong_answer1"]
    wrong_answers2 = df_results["wrong_answer2"]
    wrong_answers3 = df_results["wrong_answer3"]

    prompt = rag.base_prompt
    questions = df_results["question"]
    contexts = df_results["rag_context"]

    model, tokenizer = rag.model, rag.tokenizer

    pbar = tqdm(total=len(df_results))
    pbar.set_description("Calculating perplexity accuracy...")
    count = 0
    for i, (question, context, a, b, c, d) in enumerate(zip(questions, contexts, correct_answers, wrong_answers1, wrong_answers2, wrong_answers3)):
        perplexities_answers = [0, 0, 0, 0]

        if a is float("nan") or b is float("nan") or c is float("nan") or d is float("nan"):
            df_results.at[i, "perplexity_accuracy_score"] = float("nan")
            df_results.at[i, "perplexity_accuracy_choice"] = float("nan")
            continue

        for j, answer in enumerate([str(a), str(b), str(c), str(d)]):
            prompt_answer = build_message(tokenizer, question, [], prompt.format(text_books=context), answer)
            prompt_filled = build_message(tokenizer, question, [], prompt.format(text_books=context))

            try:
                perplexities_answers[j] = perplexity(model, tokenizer, prompt_filled, prompt_answer, device=device)
                if j > 0:
                    df_results.at[i, f"perplexity_w{j}_score"] = perplexities_answers[j]
            except:
                perplexities_answers[j] = 0

            torch.cuda.empty_cache()

        choice = np.argmin(perplexities_answers)
        df_results.at[i, "perplexity_accuracy_score"] = int(choice == 0)
        df_results.at[i, "perplexity_accuracy_choice"] = choice

        count += 1
        inc = count - pbar.n
        pbar.update(n=inc)

        pbar.set_description(f"Calculating perplexity accuracy... (mean={df_results[df_results['wrong_answer1'].notna()]['perplexity_accuracy_score'].dropna().mean():.2f})")

    return df_results
