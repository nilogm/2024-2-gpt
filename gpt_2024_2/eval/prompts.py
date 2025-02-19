EXAM_PROMPT = """Baseado no contexto seguinte, responda a última pergunta:

Aqui está o contexto:
{context}

Os itens abaixo são perguntas de múltipla escolha (com respostas).

Pergunta: Qual é a capital da França?

Opções:
- A) Londres
- B) Paris
- C) Roma
- D) Berlim

Resposta correta: B)

Pergunta: Quem escreveu o livro "Orgulho e Preconceito"?

Opções:
- A) Emily Brontë
- B) Charlotte Brontë
- C) Jane Austen
- D) Virginia Woolf

Resposta correta: C)

Pergunta: Qual é o maior rio do mundo?

Opções:
- A) Nilo
- B) Amazonas
- C) Yangtze
- D) Yangtze

Resposta correta: A)

Pergunta: Qual é o menor país do mundo?

Opções:
- A) Suriname
- B) Mônaco
- C) Luxemburgo
- D) Vaticano

Resposta correta: D)

Pergunta: Qual é a maior montanha do mundo?

Opções:
- A) Pico da Bandeira
- B) Monte Kilimanjaro
- C) Monte Evereste
- D) Monte Fuji

Resposta correta: C)

Responda a seguinte pergunta baseado no contexto dado acima no mesmo formato:
Pergunta: {question}

Opções: 
{choices}

Resposta correta: """


LLM_EXAM_PROMPT = """Determine qual alternativa é mais próxima da resposta:
Reponda SOMENTE com A, B, C ou D.
Resposta: {answer}
Opções:
{choices}

Resposta correta: """


LLM_ANALYSIS_PROMPT = """Analise e determine quais respostas são equivalentes à resposta correta da pergunta.
Para que duas respostas sejam equivalentes, elas devem dar a mesma informação ao usuário.
Para cada uma das respostas (de 1 a 5) determine se a resposta correta dá a mesma informação para o usuário, caso afirmativo, responda "<ans> SIM </ans>", caso contrário, responda "<ans> NÃO </ans>".
Mostre o seu raciocínio para cada comparação de resposta correta com cada uma das respostas (de 1 a 5).

Pergunta: {question}
Resposta correta: {answer}

Resposta 1: {a1}
Resposta 2: {a2}
Resposta 3: {a3}
Resposta 4: {a4}

Responda SOMENTE no seguinte formato:
<ans1> <veredito> </ans1>
Raciocínio: <raciocínio>
<ans2> <veredito> </ans2>
Raciocínio: <raciocínio>
<ans3> <veredito> </ans3>
Raciocínio: <raciocínio>
...

onde
<veredito> => SIM ou NÃO
<raciocínio> => explicação do motivo por qual você deu este veredito para a comparação das respostas

Agora, responda SIM ou NÃO para cada comparação e explique o seu raciocínio para cada uma:
"""
