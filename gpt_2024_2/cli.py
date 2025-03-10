import typer
from typing_extensions import Annotated
from gpt_2024_2.qa_test import test_qa
from gpt_2024_2.utils import load_env, __path_check, set_device
from gpt_2024_2.model.brain_bot import BrainBot
import gpt_2024_2.evaluate_chatbot_answers as evaluate_chatbot_answers

hf_auth = load_env()

app_experiment = typer.Typer()


def _exec_run():
    app_experiment()


@app_experiment.command(help="Initializes RAG context files")
def init(
    encoder_codename: Annotated[str, typer.Option(help="Encoder model")],
    device: str = 0,
    verbose: bool = True,
):
    device_ = set_device(device)
    model = BrainBot(generative_codename=None, encoder_codename=encoder_codename, device=device_, init=True)
    model.load()
    model.initialize_context(verbose=verbose)
    model.unload()


@app_experiment.command(help="Tests the RAG system with a Q&A dataset")
def test(
    generative_codename: Annotated[str, typer.Option(help="Generative model")],
    encoder_codename: Annotated[str, typer.Option(help="Encoder model")],
    top_k: int = 5,
    borders: int = 1,
    device: str = 0,
    qa_file: str = "qa_09_03_2025.csv",
    results_dir: str = "results",
):
    device_ = set_device(device)

    model = BrainBot(generative_codename=generative_codename, encoder_codename=encoder_codename, top_k=top_k, borders=borders, device=device_)
    model.load()
    __path_check(results_dir)
    test_qa(model, generative_codename, encoder_codename, results_dir=results_dir, test_dir="dataset/test", qa_file=qa_file)
    model.unload()


@app_experiment.command(help="Executes metrics on the results of the tests")
def eval(results_dir: str, device: str = 0):
    device_ = set_device(device)
    evaluate_chatbot_answers.evaluate(results_dir, device_)
