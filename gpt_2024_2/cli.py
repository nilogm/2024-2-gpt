import typer
import json
import os
import datetime
from gpt_2024_2.experiment import test_qa
from gpt_2024_2.utils import load_env, path_check, set_device
from gpt_2024_2.model.brain_bot import BrainBot
import gpt_2024_2.evaluation as evaluation
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

    return (
        [{"generator": i, "encoder": j, "top_k": k, "summarizer": l} for i, j, k, l in product(configs["generator"], configs["encoder"], configs["top_k"], configs["summarizer"])],
        configs["memories_path"],
        configs["memory_end_date"],
        configs["today_date"],
        configs["qa_file"],
        configs["results_dir"],
    )


@app_experiment.command(help="Executes the experiments described in the configuration file")
def experiment(config_file: str, device: str = 0, verbose: bool = True):
    params, memories_path, memory_end_date, today_date, qa_file, results_dir = read_configs(config_file)
    device_ = set_device(device)
    path_check(results_dir)

    model = BrainBot(device=device_, memories_path=memories_path, memory_end_date=datetime.datetime.strptime(memory_end_date, "%m/%d/%Y").date(), today_date=today_date)
    for p in params:
        model.setup_models(**p, verbose=verbose)
        test_qa(model, p["generator"]["nickname"], p["encoder"]["nickname"], results_dir=results_dir, qa_file=qa_file)


@app_experiment.command(help="Executes metrics on the results of the given experiment folder")
def evaluate(results_dir: str, out_dir: str, device: str = 0):
    device_ = set_device(device)
    path_check(out_dir)
    evaluation.evaluate(results_dir, out_dir, device_)
