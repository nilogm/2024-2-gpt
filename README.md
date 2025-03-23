# Memória Episódica em Modelos Generativos

Este repositório contém a implementação de uma memória episódica para modelos generativos, projetada para melhorar a capacidade de um modelo de linguagem em manter e utilizar interações passadas com o usuário. Neste trabalho, utilizamos um dataset de conversações para simular várias interações passadas entre o modelo e o usuário, permitindo que o modelo acesse e utilize informações de diálogos anteriores com base no contexto temporal e de similaridade.

## Descrição do Funcionamento

O sistema foi projetado para simular a memória episódica, ou seja, a capacidade de lembrar interações passadas em um contexto específico. Para isso, o processo é dividido nas seguintes etapas:

1. **Obtenção do Dataset de Conversações**: Utilizamos um [conjunto de dados](https://www.kaggle.com/datasets/thedevastator/dailydialog-unlock-the-conversation-potential-in?select=test.csv) de conversações entre o modelo e o usuário, onde cada diálogo é atribuído a uma data, representando o dia em que a conversa aconteceu. O dataset modificado pode ser encontrado neste mesmo repositório, em `dataset`.

2. **Geração de Embeddings**: Cada diálogo é transformado em embeddings, que são vetores numéricos que representam semanticamente o conteúdo da conversa. O processo de geração de embeddings pode ser feito de duas maneiras:
    - **Diálogos diretos**: Usando os diálogos de forma direta para gerar os embeddings.
    - **Resumo dos Diálogos**: Utilizando um método para resumir os diálogos antes de gerar os embeddings, o que pode melhorar a eficiência e relevância das informações armazenadas.

3. **Armazenamento com Faiss**: Os embeddings gerados são armazenados com Faiss, uma biblioteca altamente otimizada para pesquisa de similaridade de vetores. Isso permite uma recuperação rápida e eficiente dos diálogos relevantes durante a interação com o usuário.

4. **Processo de Resposta ao Usuário**:
    - Quando o usuário faz uma pergunta, o modelo investiga a pergunta em busca de um **período de tempo** (por exemplo, "Ontem" ou "Na semana passada") e retorna o período específico como datas interpretáveis por código.
    - O modelo então consulta a memória episódica, identificando e resgatando diálogos que ocorreram durante o período identificado, bem como diálogos semelhantes em termos de conteúdo semântico.
    - Os diálogos retornados são avaliados e apenas os que o modelo considera relevantes são selecionados.
    - Finalmente, o modelo gera a resposta ao usuário com base nos diálogos relevantes selecionados.

## Funcionalidades

- **Memória episódica**: O modelo mantém uma memória de interações passadas com base em diálogos anteriores.
- **Recuperação contextual**: O sistema é capaz de identificar o período de tempo relevante para cada pergunta do usuário.
- **Uso de Faiss para similaridade**: O modelo utiliza Faiss para realizar buscas eficientes nos embeddings gerados dos diálogos.
- **Resumo de diálogos**: Possibilidade de resumir diálogos antes de gerar embeddings para uma representação mais eficiente.

## Instalação e Setup

Para instalar as dependências do projeto, basta rodar o seguinte comando:

```bash
poetry install
```

Também é necessário inserir uma chave do [Hugging Face](https://huggingface.co) em um arquivo `sample.env` na pasta do repositório.

```bash
HUGGINGFACE_AUTH_TOKEN="[...]"
```

## Execução de Experimentos

Após isso, o comando abaixo pode ser executado para rodar os experimentos descritos no arquivo de configuração `configs.json`.

```bash
poetry run gpt experiment configs.json
```

### Arquivo `configs.json`

Este arquivo deve conter informações sobre quais experimentos devem ser executados. Portanto, os campos abaixo devem ser preenchidos.
 - **generator**: itens com informações sobre um modelo generativo. Cada item deve conter os campos "nickname" e "model_id", que representam o apelido do modelo em arquivos criados pelo código e o ID do modelo no Hugging Face.
 - **encoder**: itens com informações sobre um modelo de encoding. Cada item deve conter os campos "nickname", "model_id" e "dimensions", que representam o apelido do modelo em arquivos criados pelo código, o ID do modelo no Hugging Face, e o número de dimensões que devem ser utilizadas.
 - **top_k**: um inteiro que indica o número de itens que devem der retornados pelo banco de vetores.
 - **summarizer**: strings que representam o método de sumarização dos diálogos do dataset para realizar os encodings. Valores aceitos são: "default", "bart", "rake", "yake", "pegasus". Quaisquer outros valores serão convertidos para "default".
Cada item acima deve ser uma lista de possíveis valores, uma vez que o código irá fazer toda combinação possível de cada valor em cada campo.

Adicionalmente, é necessário informar:
 - **memories_path**: caminho para o diretório que contém os arquivos de memória (em 'csv').
 - **memory_end_date**: a data mais recente a ser artificialmente inserida no dataset.
 - **today_date**: o dia que deve ser levado em consideração como 'hoje' pelo modelo generativo.
 - **qa_file**: o dataset de perguntas e respostas que deve ser aplicado para avaliação das configurações. Um exemplo de dataset pode ser encontrado em `dataset/test/`
 - **results_dir**: diretório para onde devem ir os resultados dos testes.

Na pasta `configs/` há vários exemplos de arquivos de configuração.

## Avaliação dos Resultados

Para avaliar os experimentos, execute o comando abaixo substituindo os valores entre '<' e '>'.

```bash
poetry run gpt evaluate <diretório_de_entrada> <diretório_de_saída>
```

O notebook `results.ipynb` apresenta exemplos de como fazer gráficos com os resultados obtidos.