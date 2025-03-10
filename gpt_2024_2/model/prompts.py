# TODO: adaptar esse prompt para que ele tenha ciência de tempo.
CHAT_PROMPT = """Você é um assistente inteligente. Interaja com o usuário com base nas conversas dadas e na data de hoje: {today}.
Se a pergunta que o usuário fez não está relacionada com uma conversa, desconsidere tal conversa. Não mencione nada que não aparece nas conversas.
Ao fim, analise a sua resposta para ter certeza que sua resposta está de acordo com o que é mencionado nas conversas.

Aqui estão as conversas relacionadas:
{conversations}

Usuário: """


DATE_PROMPT = """Você irá receber uma mensagem de um usuário, identifique a data mencionada pela mensagem com base na data de hoje.
Leia a mensagem do usuário e retorne um JSON contento o período ao que o usuário de refere.
Não responda a mensagem do usuário, somente analise a mensagem para identificar uma data e retorne o JSON.
O JSON deve ser formatado da seguinte forma {{ "dates" : [ <itens> ] }}, onde
 - Cada item em <itens> deve estar no formato {{ "start": <início_do_período>, "end": <fim_do_período> }}
 - <início_do_período> marca o dia em que começa o período citado pela mensagem
 - <fim_do_período> marca o dia em que o período termina (se for um período de mais de um dia)

Exemplos:
Data: 09/03/2025
Usuário: "Você viu o que o Pedro fez ontem?"
Resposta: {{ "dates" : [ {{ "start" : "08/03/2025" }} ] }}

Data: 03/03/2025
Usuário: "Tinha algum dever para semana passada? e para a anterior?"
Resposta: "de 23/02/2025 até 01/03/2025 e 16/02/2025 até 22/02/2025"
Resposta: {{ "dates" : [ {{ "start" : "23/02/2025", "end" : "01/03/2025" }}, {{ "start" : "16/02/2025", "end" : "22/02/2025" }} ] }}

Data: 12/03/2025
Usuário: "Já te contei esta história antes?"
Resposta: {{ "dates" : [] }}

Data: 12/03/2025
Usuário: "Quando falamos disso?"
Resposta: {{ "dates" : [] }}

Data: 09/03/2025
Usuário: "falamos disso há algumas semanas."
Resposta: {{ "dates" : [ {{ "start": "23/02/2025", "end": "08/03/2025" }} ] }}

Sua vez:
Data de hoje: {today}
Usuário: """


from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def build_message(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, message: str, system_prompt: str):
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if message:
        messages.append({"role": "user", "content": message})

    if not tokenizer:
        return messages

    # add_generation_prompt=True fornece o começo da próxima resposta (ex.:<|start_header_id|>assistant<|end_header_id|>)
    formatted_message = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return formatted_message
