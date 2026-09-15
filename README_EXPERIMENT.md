# Protótipo: RNN com memória relacional distribuída + AI-Memory

## Objetivo

Validar, em um experimento pequeno e controlado, a ideia de não comprimir todo o passado no estado oculto da RNN. O protótipo separa:

1. **Estado recorrente** — controlador neural de curto prazo.
2. **Memória relacional interna distribuída por camadas** — fatos estruturados com `fact_id` global.
3. **AI-Memory externa** — memória de projeto/contexto atual, usando `DimensionalMemoryFinal` / V3 Fast.
4. **Copy/read path** — permite gerar diretamente um valor recuperado sem obrigá-lo a atravessar uma compressão adicional no estado oculto.

## Arquitetura testada

Cada fato mantém a forma simbólica:

```text
(fact_id, subject_id, relation_id, value_id)
```

O mesmo `fact_id` ocupa linhas alinhadas nas memórias das duas camadas.
Cada camada calcula sua própria representação Transformer-like:

```text
Layer 1: K1[fact_id], V1[fact_id]
Layer 2: K2[fact_id], V2[fact_id]
```

As duas camadas possuem visões completas do fato, mas projeções diferentes. O fato não é arbitrariamente quebrado entre camadas.

A consulta produz escores de atenção separados:

```text
a1 = attention(q1, K1)
a2 = attention(q2, K2)
```

A leitura final exige concordância entre as matrizes sobre o mesmo `fact_id`:

```text
joint = normalize(a1 * a2)
read  = joint @ V2
```

As matrizes K/V são construídas **uma vez por sequência** e reutilizadas em todos os passos recorrentes.

Para fatos relacionais, a distribuição `joint` também alimenta um caminho de cópia para `value_id`. Isso evita recuperar o fato corretamente e depois perder precisão ao recomprimi-lo no estado oculto.

## Segunda memória: AI-Memory

O protótipo usa a interface:

```python
from dimensional_memory_final import DimensionalMemoryFinal
```

Foi reproduzido no sandbox o subconjunto necessário da implementação V3 Fast do repositório `Paullovitt/AI-Memory`, porque o ambiente não possui acesso DNS direto ao GitHub durante a execução.

A AI-Memory contém registros de projetos, por exemplo:

```text
projeto P49 usa linguagem Python banco MySQL
```

A busca externa retorna o registro relevante. Um pequeno encoder bidirecional contextualiza somente esse registro recuperado e o modelo pode copiar o token necessário para a geração.

## Tarefa de treino

O modelo gera autoregressivamente sequências como:

```text
resposta V20 . <eos>
```

ou, usando a memória de projeto:

```text
resposta Python . <eos>
```

Durante este primeiro experimento, `(entidade, relação)` é entregue de forma estruturada ao controlador. Isso isola o teste da memória em relação a NER/parsing de linguagem natural.

## Modelos comparados

| Modelo | Parâmetros |
|---|---:|
| GRU baseline, 2 camadas | 213.798 |
| RNN + memória distribuída + AI-Memory | 200.049 |

A baseline recebe todos os fatos serializados como sequência. O modelo experimental recebe a pergunta curta e consulta a memória estruturada.

## Resultado principal

| Nº de fatos | Baseline — valor correto | Memória — valor correto | fact_id top-1 |
|---:|---:|---:|---:|
| 12 | 2,81% | **100%** | **100%** |
| 32 | 2,19% | **100%** | **100%** |
| 64 | 2,81% | **100%** | **100%** |
| 128 | 2,19% | **100%** | **100%** |
| 256 | 3,75% | **100%** | **100%** |
| 512 | 4,17% | **100%** | **100%** |

Os testes de 256 e 512 fatos foram feitos **sem retreino**.

### Latência CPU no stress test

| Nº de fatos | Baseline | Memória distribuída |
|---:|---:|---:|
| 128 | 4,65 ms/amostra | **2,03 ms/amostra** |
| 256 | 11,35 ms/amostra | **3,67 ms/amostra** |
| 512 | 31,38 ms/amostra | **8,56 ms/amostra** |

Esses valores são específicos do sandbox e não constituem benchmark de hardware geral.

## Exemplo real do teste

```text
Pergunta: pergunta qual valor de E16 relacao R5 ?
Esperado: resposta V20 . <eos>
Baseline: resposta Java . <eos>
Memória:  resposta V20 . <eos>
```

Memória de projeto, usando projeto não visto no treino do gerador:

```text
Pergunta: pergunta qual linguagem usa projeto P49 ?
Esperado: resposta Python . <eos>
Memória:  resposta Python . <eos>
```

## O que este teste demonstra

- O estado recorrente não precisa carregar todos os fatos.
- IDs globais funcionam bem para alinhar diferentes representações do mesmo fato entre camadas.
- Matrizes K/V por camada permitem comparação e recuperação Transformer-like sem guardar o histórico textual inteiro.
- O caminho de cópia preserva valores exatos depois da recuperação.
- A AI-Memory funciona como uma memória externa separada para projeto/contexto ativo.
- No cenário sintético, aumentar a quantidade de fatos de 12 para 512 não degradou a recuperação relacional.

## O que ainda NÃO demonstra

Este resultado não prova desempenho equivalente em linguagem natural geral. O benchmark é deliberadamente sintético e controlado.

Ainda faltam:

- aprender automaticamente a extração entidade/relação;
- fatos conflitantes e versionamento temporal;
- atualização e remoção de fatos durante a geração;
- entidades ambíguas e aliases;
- relações multi-hop;
- memória distribuída com milhões de fatos;
- recuperação aproximada/semântica quando os termos não coincidem;
- comparação contra Transformer/RAG e arquiteturas modernas de memória;
- ablação formal de 1 camada vs 2+ camadas de memória;
- testes em corpus textual real.

## Arquivos

- `experiment.py` — treino, geração, baseline e modelo experimental.
- `stress_test.py` — extrapolação para 128/256/512 fatos.
- `results.json` — métricas do treino e testes principais.
- `stress_results.json` — métricas de extrapolação e latência.
- `baseline_gru.pt` — pesos da baseline.
- `distributed_memory_rnn.pt` — pesos do modelo experimental.
- `ai-memory/` — subconjunto da implementação usada da AI-Memory.

## Reproduzir

```bash
python experiment.py
python stress_test.py
```

Requer Python e PyTorch. A implementação de memória copiada para `ai-memory/` não requer dependências extras além da biblioteca padrão.
