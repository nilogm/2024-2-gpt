# GPT 2024/2

## Algumas coisas que faltam implementar:
 - [ ] Adaptar o prompt para incluir noção de tempo
 - [ ] Atribuir datas para as conversas
 - [ ] Fazer um dataset de perguntas para teste
 - [ ] Adicionar função para recuperar conversas com base na data
 - [ ] Encontrar um método de avaliação
 - [ ] Encontrar modelos generativos e encoders para teste
 - [ ] Encontrar parametrizações
 - [ ] Propor uma maneira de fazer o modelo generativo entender o conceito de tempo
 - [ ] Adaptar o código de retrieval comum para levar em consideração índices próximos na mesma conversa
 - [ ] Criar um README.md para o repositório

### Possíveis parametros
 - Número de conversas para retornar
 - Número de mensagens para retornar antes e depois da mensagem mais similar encontrada

### Adendos
 - Talvez seja bom fazer embeddings considerando a mensagme anterior e a próxima (ou 2 antes, etc.) para que haja algum tipo de contexto da conversa para uma melhor busca por similaridade.
