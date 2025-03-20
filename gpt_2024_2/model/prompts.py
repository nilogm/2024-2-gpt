CHAT_PROMPT = """You're an intelligent assistant. Interact with the user based on the previous conversations and today's date: {today}.
If the user's question is not related to a conversation, do not consider that conversation when answering the user. Do not mention anything that doesn't appear in the conversations.
Finally, analyze your answer to be sure your answer makes sense given the conversations below.

Here are the previous conversations between you and the user:
{conversations}

User: """


DATE_PROMPT = """You'll receive a message from a user. Identify the date mentioned by the message based on today's date.
Read the user's message and return a JSON with the period of time the user is refering to. If no time period is specified, DO NOT MAKE ONE UP.
Do not answer the user's message, only analyze the message to identify the time period it refers to and return the JSON.
The JSON must be formatted in the following format: {{ "dates" : [ <items> ] }}, where
 - Each item in <items> must be in the format {{ "start": <start_date>, "end": <end_date> }}
 - <start_date> marks the day in which the mentioned period starts
 - <end_date> marks the day in which the mentioned period ends (if the period mentioned is longer than a day)

Examples:
Date: 03/09/2025
User: "Did you see what Pedro did yesterday?"
Notes: The message asks if something happened YESTERDAY.
Answer: {{ "dates" : [ {{ "start" : "03/08/2025" }} ] }}

Date: 03/03/2025
User: "Was there homework due last week? What about the week before?"
Notes: The message asks if something happened LAST WEEK. Then, it asks if something happened the WEEK BEFORE that.
Answer: {{ "dates" : [ {{ "start" : "02/23/2025", "end" : "03/01/2025" }}, {{ "start" : "02/16/2025", "end" : "02/22/2025" }} ] }}

Date: 03/12/2025
User: "Have I told yout that story before?"
Notes: The message does not mention a time period.
Answer: {{ "dates" : [] }}

Date: 03/12/2025
User: "When did we talk about this?"
Notes: The message does not mention a time period.
Answer: {{ "dates" : [] }}

Date: 03/09/2025
User: "We talked about that some weeks ago."
Notes: The message mentions 'SOME WEEKS AGO', which could mean a couple of weeks ago.
Answer: {{ "dates" : [ {{ "start": "02/23/2025", "end": "03/08/2025" }} ] }}

Date: 09/03/2025
User: "Rembember when I did this?"
Notes: No specific time period was mentioned.
Answer: {{ "dates" : [] }}

Your turn!
Today's date: {today}
User: """


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
