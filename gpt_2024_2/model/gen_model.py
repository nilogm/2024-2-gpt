import transformers
import torch
from typing import Tuple, Any
from transformers.pipelines import Pipeline
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def init_model(model_id, hf_auth, verbose=False, device_map=0) -> Tuple[Pipeline, Any, PreTrainedTokenizer | PreTrainedTokenizerFast]:
    if verbose:
        print("Initializing LLM model...")
        print(model_id)

    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_auth,
        device_map=device_map,
        device=device_map,
        padding_side="left",
    )

    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_auth,
        device_map=device_map,
        device=device_map,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_auth,
        config=model_config,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    model.eval()

    deterministic = dict(
        do_sample=False,
        num_return_sequences=1,
        temperature=None,
        top_p=None,
    )

    # Initialize pipeline
    model_pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
        device_map=device_map,
        max_new_tokens=1024,  # max number of tokens to generate in the output
        repetition_penalty=1.1,
        **deterministic,
        batch_size=1,
    )

    model_pipeline.tokenizer.pad_token_id = model.config.eos_token_id if isinstance(model.config.eos_token_id, int) else model.config.eos_token_id[0]
    return model_pipeline, model, tokenizer
