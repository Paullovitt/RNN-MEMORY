# RNN-MEMORY

Protótipo experimental de uma **RNN/GRU com memória relacional distribuída por camadas**, memória externa de projeto e caminhos de cópia para preservar valores exatos.

O objetivo é testar uma hipótese: em vez de obrigar a rede recorrente a comprimir todo o histórico em um único estado oculto `h_t`, separar **processamento**, **memória relacional** e **memória de contexto/projeto**.

> **Status:** pesquisa/prova de conceito. Os benchmarks atuais são sintéticos e controlados. Eles validam o mecanismo implementado neste repositório, mas não demonstram desempenho geral em linguagem natural nem superioridade sobre Transformers modernos.

## Arquitetura

```text
                         +----------------------+
Pergunta --------------> | Controlador RNN/GRU  |
                         +----------+-----------+
                                    |
                  +-----------------+-----------------+
                  |                                   |
                  v                                   v
        Memória relacional interna             AI-Memory externa
        distribuída por camadas                projeto/contexto
                  |                                   |
                  +-----------------+-----------------+
                                    |
                                    v
                            leitura / copy path
                                    |
                                    v
                              geração da saída
```

### Fatos relacionais

Cada fato é representado de forma estruturada:

```text
(subject_id, relation_id, value_id, fact_id)
```

Exemplo conceitual:

```text
(João, mora_em, Curitiba, F9281)
(João, possui, Rex,      F9282)
(Rex,  idade,    4,      F9283)
```

O `fact_id` identifica o fato globalmente. No protótipo atual ele é usado para acompanhar e diagnosticar a identidade do fato; o cálculo neural usa `subject`, `relation` e `value`.

O fato **não é quebrado arbitrariamente entre camadas**. Cada camada mantém uma visão completa, mas com projeções diferentes.

### Memória distribuída por camadas

O modelo usa duas representações K/V alinhadas pela mesma linha de fato:

```text
camada 1 -> K1[i], V1[i]
camada 2 -> K2[i], V2[i]
```

As chaves usam entidade + relação:

```text
K1 = key1(subject_embedding || relation_embedding)
K2 = key2(subject_embedding || relation_embedding)
```

Os valores recebem o fato completo:

```text
V1 = value1(subject || relation || value)
V2 = value2(subject || relation || value)
```

Cada camada calcula sua própria atenção:

```text
a1 = softmax(score(q1, K1))
a2 = softmax(score(q2, K2))
```

A leitura final exige concordância entre as duas matrizes:

```text
joint = normalize(a1 * a2)
read  = joint @ V2
```

As matrizes K/V são construídas **uma vez por sequência** e reutilizadas durante os passos recorrentes.

### Estado recorrente como controlador

O modelo possui duas `GRUCell`. O estado oculto influencia a consulta:

```text
q1 = normalize(query_key_1 + hidden_projection_1(h1))
q2 = normalize(query_key_2 + hidden_projection_2(h2, read1))
```

A divisão de responsabilidade é:

```text
estado h_t      -> processamento e controle de curto prazo
memória K/V     -> fatos relacionais explícitos
AI-Memory       -> contexto/projeto externo recuperável
```

### Copy path

Depois que `joint` seleciona os fatos relevantes, a distribuição pode ser projetada diretamente sobre os `value_id`.

```text
fact attention
      |
      v
 value_id distribution
      |
      v
 geração
```

Isso evita recuperar corretamente um fato e depois perder precisão ao recomprimi-lo no estado oculto.

A saída mistura três fontes:

```text
1. geração normal pelo estado recorrente
2. cópia da memória relacional
3. cópia do contexto recuperado pela AI-Memory
```

Um gate aprendido decide quanto usar de cada fonte.

## Segunda memória: AI-Memory

O diretório `ai-memory/` contém a parte necessária da memória externa desenvolvida em:

https://github.com/Paullovitt/AI-Memory

Ela funciona como memória de projeto/contexto atual.

Exemplo:

```text
projeto P49 usa linguagem Python banco MySQL
```

Fluxo:

```text
pergunta sobre projeto
        |
        v
AI-Memory.search(...)
        |
        v
registro relevante
        |
        v
encoder contextual curto
        |
        v
atenção/cópia do token necessário
```

## Treinamento atual

O dataset atual é sintético para isolar o comportamento da memória.

Consulta relacional:

```text
pergunta qual valor de E16 relacao R5 ?
```

Saída:

```text
resposta V20 . <eos>
```

Consulta de projeto:

```text
pergunta qual linguagem usa projeto P49 ?
```

Saída:

```text
resposta Python . <eos>
```

Neste estágio, `subject` e `relation` também são fornecidos de forma estruturada. Portanto, o benchmark **não mede NER, parsing ou linguagem natural aberta**. Ele mede principalmente recuperação, recorrência, copy path e escala de memória.

## Baseline

| Modelo | Parâmetros |
|---|---:|
| GRU baseline | 213.798 |
| RNN + memória distribuída + AI-Memory | 200.049 |

A baseline recebe todos os fatos serializados na sequência. O modelo experimental recebe uma pergunta curta e consulta fatos estruturados fora do estado oculto.

## Resultados reproduzíveis

| Nº de fatos | Baseline: valor correto | Modelo com memória | `fact_id` top-1 |
|---:|---:|---:|---:|
| 12 | 2,81% | **100%** | **100%** |
| 32 | 2,19% | **100%** | **100%** |
| 64 | 2,81% | **100%** | **100%** |
| 128 | 2,19%* | **100%** | **100%** |
| 256 | 3,75% | **100%** | **100%** |
| 512 | 4,17% | **100%** | **100%** |

`*` O teste principal de 128 fatos em `results.json` mede 2,19%; a execução separada em `stress_results.json`, com outra amostra sintética, mede 3,75% para a baseline.

Os testes de 256 e 512 fatos foram feitos sem retreinar o modelo para esses tamanhos.

### Latência CPU

| Nº de fatos | Baseline | Modelo com memória |
|---:|---:|---:|
| 128 | 4,65 ms/amostra | **2,03 ms/amostra** |
| 256 | 11,35 ms/amostra | **3,67 ms/amostra** |
| 512 | 31,38 ms/amostra | **8,56 ms/amostra** |

Esses tempos são específicos do ambiente usado no benchmark.

## Estrutura do repositório

```text
RNN-MEMORY/
|
|-- experiment.py
|   treino, dataset sintético, baseline e modelo com memória
|
|-- stress_test.py
|   extrapolação para 128 / 256 / 512 fatos
|
|-- ai-memory/
|   segunda memória externa
|
|-- distributed_memory_rnn.pt
|   checkpoint do modelo experimental
|
|-- results.json
|   métricas do experimento principal
|
|-- stress_results.json
|   métricas de extrapolação e latência
|
|-- requirements.txt
|
`-- README_EXPERIMENT.md
    relatório inicial
```

`baseline_gru.pt` não está versionado atualmente. Ele é criado por `experiment.py` e depois usado por `stress_test.py`.

## Instalação

```bash
git clone https://github.com/Paullovitt/RNN-MEMORY.git
cd RNN-MEMORY
pip install -r requirements.txt
```

Requer Python 3.10+ e PyTorch (`torch>=2.0`).

## Reproduzir

Primeiro:

```bash
python experiment.py
```

Esse comando treina os modelos, executa os testes principais, grava `results.json`, cria `baseline_gru.pt` e atualiza `distributed_memory_rnn.pt`.

Depois:

```bash
python stress_test.py
```

> Em um clone novo, execute `experiment.py` antes de `stress_test.py`, pois o stress test precisa de `baseline_gru.pt`.

## O que o protótipo demonstra

Dentro do cenário sintético atual:

- o estado recorrente não precisa armazenar sozinho todos os fatos;
- diferentes camadas podem manter projeções completas do mesmo conjunto de fatos;
- o alinhamento das linhas permite combinar as atenções das camadas;
- recuperação e geração podem ser separadas;
- o copy path preserva valores exatos;
- uma memória externa separada pode fornecer contexto de projeto;
- os testes versionados não apresentaram degradação de recuperação até 512 fatos.

## Limitações atuais

Ainda não foi demonstrado de forma geral:

- linguagem natural aberta;
- extração automática robusta de entidades e relações;
- entidades ambíguas e aliases;
- fatos contraditórios;
- versionamento temporal;
- atualização/remoção dinâmica de fatos;
- raciocínio multi-hop;
- recuperação semântica sem correspondência lexical;
- milhões de fatos em benchmark reproduzível versionado;
- geração longa;
- comparação formal contra Transformer, RAG, Mamba e outras arquiteturas;
- escrita end-to-end na memória.

Os testes maiores devem entrar no repositório junto com scripts e resultados reproduzíveis antes de serem tratados como benchmark oficial.

## Próximos experimentos

1. aprender automaticamente entidade e relação a partir de texto;
2. permitir escrita, atualização e invalidação dinâmica de fatos;
3. adicionar tempo, versão e confiança;
4. testar relações multi-hop;
5. indexar a memória para evitar varredura completa em escalas grandes;
6. testar milhões de fatos;
7. fazer ablação de 1, 2 e múltiplas camadas de memória;
8. comparar contra Transformer/RAG sob o mesmo dataset e orçamento;
9. testar corpus textual real e geração longa.

## Resumo

```text
estado recorrente pequeno        -> controle
memória relacional distribuída   -> fatos
AI-Memory externa                -> projeto/contexto
copy paths                       -> precisão
                                  |
                                  v
                               geração
```

A hipótese do projeto é que **processar informação e armazenar informação não precisam ser a mesma operação**.
