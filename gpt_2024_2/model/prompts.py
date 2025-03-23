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
Please, format the dates only as following: month/day/year. And please, be sure that the date you mention corresponds to the weekday the user mentions, if any.

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
User: "Have I told you that story before?"
Notes: The message does not mention a time period.
Answer: {{ "dates" : [] }}

Date: 03/12/2025
User: "When did we talk about this?"
Notes: The message does not mention a time period.
Answer: {{ "dates" : [] }}

Date: 03/09/2025
User: "We talked about that some weeks ago."
Notes: The message mentions 'SOME WEEKS AGO', which could mean a couple of weeks ago.
Answer: {{ "dates" : [ {{ "start": "02/23/2025", "end": "03/01/2025" }} ] }}

Date: 09/03/2025
User: "What was I doing a year ago?"
Notes: Retrieve only a short period, as a year contains many memories.
Answer: {{ "dates" : [ {{ "start": "03/07/2024", "end": "03/11/2024" }} ] }}

Date: 09/03/2025
User: "Rembember when you said this?"
Notes: No specific time period was mentioned, so no dates should be extracted.
Answer: {{ "dates" : [] }}

Your turn! Return only the JSON with the dates.
Today's date: {today}
User: """


RELEVANCY_PROMPT = """You are going to receive some conversations between you and the user. Determine which conversations are relevant to answer the user's question givent their content and date.
At the end, return only a JSON with the indices of the relevant conversations.
Examples:
 - if only conversations 2, 6 and 7 are necessary to answer the question, answer only with "{{ "relevant_conversations": [2, 6, 7] }}"
 - if no conversation is necessary to answer the question, answer only with "{{ "relevant_conversations": [] }}"
 
These are the conversations, with each one containing an index and the day they happened:
{conversations}


Now, consider that today is {today}. Read the user's message below and determine which conversations are needed to respond to the user's message. Please respond only with the mentioned JSON format.
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
