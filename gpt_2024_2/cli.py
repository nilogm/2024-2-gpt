import typer
import json
import os
from gpt_2024_2.qa_test import test_qa
from gpt_2024_2.utils import load_env, path_check, set_device
from gpt_2024_2.model.brain_bot import BrainBot
import gpt_2024_2.evaluate_chatbot_answers as evaluate_chatbot_answers
from itertools import product

hf_auth = load_env()

app_experiment = typer.Typer()


def _exec_run():
    app_experiment()


def read_configs(config_file: str):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file could not be found! Path given: {config_file}")

    with open(config_file) as f:
        configs = json.load(f)

    return product(configs["generator_codename"], configs["encoder_codename"], configs["top_k"], configs["summarizer"])


@app_experiment.command(help="Tests the RAG system with a Q&A dataset")
def test(
    config_file: str,
    device: str = 0,
    verbose: bool = True,
    qa_file: str = "dataset/test/qa.csv",
    results_dir: str = "results",
):
    device_ = set_device(device)
    path_check(results_dir)

    model = BrainBot(device=device_)
    for generator, encoder, top_k, summarizer in read_configs(config_file):
        print(generator, encoder, top_k, summarizer)
        model.setup_models(generator, encoder, top_k=top_k, summarizer=summarizer)
        # test_qa(model, generator[0], encoder[0], results_dir=results_dir, qa_file=qa_file)


@app_experiment.command(help="Executes metrics on the results of the tests")
def eval(results_dir: str, out_dir: str, device: str = 0):
    device_ = set_device(device)
    path_check(out_dir)
    evaluate_chatbot_answers.evaluate(results_dir, out_dir, device_)
