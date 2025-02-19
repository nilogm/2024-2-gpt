import os
from dotenv import load_dotenv
from pathlib import Path


models = {"llama": ("meta-llama/Llama-3.1-8B-Instruct"), "glm": ("THUDM/glm-4-9b-chat"), "deepseek": ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")}
encoders = {"gte": ("thenlper/gte-large", 1024, "gte"), "jina": ("jinaai/jina-embeddings-v3", 1024, "jina"), "multilingual": ("intfloat/multilingual-e5-large-instruct", 1024, "multilingual")}


def __path_check(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return Path(dir)


def load_env():
    env_file = Path(os.getcwd()).joinpath("sample.env")
    load_dotenv(env_file)

    return os.getenv("HUGGINGFACE_AUTH_TOKEN")


def get_model(generative_codename: str):
    if generative_codename not in models:
        print(f"Unrecognized model codename: {generative_codename}. No model will be loaded!")
        return None

    return models[generative_codename]


def get_encoder(encoder_codename: str):
    if encoder_codename not in encoders:
        print(f"Unrecognized encoder codename: {encoder_codename}. No encoder will be loaded!")
        return None, None, None

    encoder_model_id, encoder_model_dimensions, encoder_path = encoders[encoder_codename]

    data_dir = f"data/{encoder_path}"
    data_path = __path_check(data_dir)

    return encoder_model_id, encoder_model_dimensions, data_path
