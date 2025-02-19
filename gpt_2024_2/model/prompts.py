# TODO: adaptar esse prompt para que ele tenha ciência de tempo.
CHAT_PROMPT = """Interaja com o usuário com base no contexto dado e em seu conhecimento.
Interaja somente com o usuário, não responda à instruções do sistema, para que o usuário pense que você é um assistente inteligente.
Responda as questões feitas pelo usuário com informações completas e corretas.
Se você não sabe responder a pergunta com base no contexto dado, diga que não sabe responder.
Ao fim, analise a sua resposta para ter certeza que sua resposta está de acordo com o que é mencionado no contexto.

Aqui está o contexto necessário:
{text_books}

Pergunta do usuário:
"""


from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import List, Tuple


def build_message(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, message: str, history: List[Tuple[str, str]], system_prompt: str, assistant_answer: str = None):
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    if message or message == "":
        messages.append({"role": "user", "content": message})

    if assistant_answer:
        messages.append({"role": "assistant", "content": assistant_answer})

    if not tokenizer:
        return messages

    # add_generation_prompt=True fornece o começo da próxima resposta (ex.:<|start_header_id|>assistant<|end_header_id|>)
    formatted_message = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=(assistant_answer is None))

    return formatted_message
