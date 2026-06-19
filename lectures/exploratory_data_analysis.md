---
title: Análise Exploratória de Dados como Investigação Estatística 
author: Diogo G. Bonofre dos Santos
---

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Introdução](#introduo)
- [Por quê EDA existe?](#por-qu-eda-existe)
- [Problemas que a EDA tenta mitigar](#problemas-que-a-eda-tenta-mitigar)
  - [Dados são incompletos](#dados-so-incompletos)
  - [Dados são codificados, não autoexplicativos](#dados-so-codificados-no-autoexplicativos)
  - [Estatísticas resumidas podem enganar](#estatsticas-resumidas-podem-enganar)
  - [Padrões visuais são hipóteses, não conclusões](#padres-visuais-so-hipteses-no-concluses)
  - [Modelar sem EDA é perigoso](#modelar-sem-eda--perigoso)
- [Contexto Histórico](#contexto-histrico)
- [Classificação de Dados Estatísticos](#classificao-de-dados-estatsticos)
- [Classificação de Dados Computacional](#classificao-de-dados-computacional)
- [Dataset Titanic](#dataset-titanic)
  - [Estrutura](#estrutura)
- [Referências](#referncias)
- [Bibliografia](#bibliografia)

<!-- markdown-toc end -->

# Introdução
A Análise Exploratória de Dados existe porque datasets reais não nascem prontos para modelagem.

Eles contêm:

- Observações incompletas;
- Erros de medição;
- Pressupostos ocultos;
- Variáveis ambíguas;
- Amostras enviesadas;
- Outliers;
- Tipos de dados misturados;
- Sumarizações enganosas;
- Contexto social, histórico ou operacional.

Portanto, o objetivo da EDA não é meramente “visualizar dados”. O objetivo é reduzir a incerteza sobre nossas premissas antes de modelarmos a solução.

# Por quê EDA existe?
Antes de modelar, precisamos entender que tipo de objeto estamos analisando.

Um dataset não é meramente uma tabela. Ele é resultado de um processo:

1. Fenômeno do mundo real
2. Medição/registro
3. Representação dos dados
4. Limpeza e transformação
5. Análise
6. Modelagem
7. Decisão

Em cada etapa, informação pode ser perdida, distorcida, simplificada ou enviesada.

# Problemas que a EDA tenta mitigar

## Dados são incompletos
Alguns valores estão ausentes porque não foram coletados, não se aplicam, foram perdidos, censurados ou omitidos intencionalmente.

Exemplo:

- A idade pode estar ausente para alguns passageiros.
- A informação da cabine pode estar ausente para muitos passageiros.

**Questão**: A ausência é aleatória ou carrega informação?

## Dados são codificados, não autoexplicativos
Variáveis podem parecer simples, mas esconder complexidade semântica.

**Exemplo**:

```python
pclass = 1, 2, 3
```
Isso é armazenado numericamente, mas não é uma variável _numérica contínua_. É uma variável _categórica ordinal_ representando a classe do bilhete.

**Atenção**: O tipo de dado em memória nem sempre é o mesmo que o tipo estatístico.

## Estatísticas resumidas podem enganar
Média, mediana, desvio-padrão e correlação são úteis, mas comprimem a realidade.

- Uma média pode esconder assimetria.
- Uma correlação pode esconder subgrupos.
- Uma tabela limpa pode esconder contexto histórico ou social.

## Padrões visuais são hipóteses, não conclusões
Em relação ao dataset Titanic, um gráfico pode sugerir:

- Mulheres tiveram taxas maiores de sobrevivência.
- Passageiros da primeira classe tiveram taxas maiores de sobrevivência.
- Crianças podem ter tido padrões diferentes de sobrevivência.

Mas isso ainda não são afirmações causais.

A EDA nos ajuda a formar hipóteses. Ela não as prova por si só.

## Modelar sem EDA é perigoso

Sem EDA, podemos:

- Usar o tipo errado de variável;
- Imputar valores ausentes incorretamente;
- Codificar categorias de forma ruim;
- Treinar com vazamento de dados;
- Ignorar desbalanceamento de classes;
- Confiar demais em correlações;
- Deixar de perceber estruturas de subgrupos.

Isso cria uma boa transição:

EDA é a ponte entre dados brutos e modelagem responsável.

# Contexto Histórico
A estatística clássica frequentemente se concentrou em inferência formal: estimar parâmetros, testar hipóteses e tirar conclusões sobre populações a partir de amostras.

A Análise Exploratória de Dados, fortemente associada a _John Tukey_, deslocou a atenção para o ato de olhar diretamente para os dados: encontrar padrões, anomalias, estruturas e perguntas antes de se comprometer com um modelo formal.

Na ciência de dados moderna, a EDA cumpre um papel semelhante: ela nos ajuda a entender o dataset antes do pré-processamento, da engenharia de atributos, da seleção de modelos e da avaliação.

> A inferência formal pergunta: _“O que podemos concluir?”_, a EDA pergunta primeiro: _“O que estamos olhando?”_.

# Classificação de Dados Estatísticos
- **Numérico**: Informação expressa em uma escala numérica.
    - **Contínuo**: Dado que pode assumir qualquer valor dentro de um intervalo.  
        - *Exemplos*: altura, peso, temperatura, preço, tempo de execução.  
        - *Sinônimos*: intervalo, flutuante, quantitativo contínuo.
    - **Discreto**: Dado descrito por valores inteiros, geralmente associado a contagens.  
        - *Exemplos*: número de filhos, quantidade de compras, número de acessos.  
        - *Sinônimos*: inteiro, contagem, quantitativo discreto.

- **Categórico**: Dado que assume valores dentro de um conjunto limitado de categorias.
    - **Nominal**: Categoria sem ordenamento natural explícito.  
        - *Exemplos*: cor, cidade, tipo de produto, gênero de filme.  
        - *Sinônimos*: fator, enumeração, classe.
    - **Binário**: Caso especial de dado categórico que assume apenas uma de duas categorias.  
        - *Exemplos*: sim/não, verdadeiro/falso, 0/1, aprovado/reprovado.  
        - *Sinônimos*: dicotômico, lógico, booleano, indicador.
    - **Ordinal**: Categoria com ordenamento explícito, mas sem distância necessariamente uniforme entre os níveis.  
        - *Exemplos*: ruim/regular/bom/ótimo, pequeno/médio/grande, 1º/2º/3º.  
        - *Sinônimos*: fator ordenado, escala ordinal.

# Classificação de Dados Prática para EDA
- **Temporal**: Informação associada ao tempo, podendo representar datas, horários, durações ou sequências temporais.
  - **Data**: Representa um dia específico no calendário.  
    - *Exemplos*: data de nascimento, data da compra, data de cadastro.
  - **Hora / Timestamp**: Representa um instante específico no tempo, geralmente com hora, minuto e segundo.  
    - *Exemplos*: horário de login, timestamp de transação, momento de coleta.
  - **Duração**: Representa um intervalo de tempo decorrido.  
    - *Exemplos*: tempo de sessão, tempo de entrega, tempo até falha.
  - **Série temporal**: Observações organizadas em sequência ao longo do tempo.  
    - *Exemplos*: vendas diárias, temperatura por hora, acessos mensais.

- **Texto / Não estruturado**: Informação representada em linguagem natural ou formato livre, geralmente exigindo pré-processamento antes da análise estatística.
  - **Texto curto**: Fragmentos pequenos de linguagem.  
    - *Exemplos*: título de produto, nome de categoria, assunto de e-mail.
  - **Texto longo**: Conteúdo textual mais extenso.  
    - *Exemplos*: comentários, reviews, descrições, mensagens.
  - **Texto semiestruturado**: Texto com algum padrão interno, mas ainda não diretamente tabular.  
    - *Exemplos*: logs, respostas de formulários, registros clínicos, descrições com códigos.

- **Geoespacial**: Informação associada a localização, posição ou distância no espaço.
  - **Coordenadas**: Representação numérica direta de localização.  
    - *Exemplos*: latitude, longitude, altitude.
  - **Região / Localidade**: Representação categórica de uma localização.  
    - *Exemplos*: país, estado, cidade, bairro, CEP.
  - **Medidas espaciais**: Variáveis derivadas de relações geográficas.  
    - *Exemplos*: distância até um hospital, área de uma região, densidade populacional.

- **Identificadores**: Variáveis usadas para identificar entidades, registros ou transações, mas que normalmente não possuem significado estatístico direto.
  - **Identificador único**: Valor usado para distinguir uma entidade individual.  
    - *Exemplos*: id_cliente, id_produto, número do pedido, matrícula.
  - **Código administrativo**: Código usado por sistemas ou instituições para referência.  
    - *Exemplos*: CPF, CNPJ, SKU, código de município.
  - **Chave relacional**: Campo usado para conectar tabelas diferentes.  
    - *Exemplos*: user_id, order_id, transaction_id.

- **Dados compostos / estruturados**: Dados armazenados em formatos internos mais complexos, frequentemente contendo múltiplas informações dentro de uma única célula.
  - **Lista**: Conjunto de múltiplos valores associados a uma observação.  
    - *Exemplos*: lista de produtos comprados, tags de um artigo, sintomas relatados.
  - **Dicionário / JSON**: Estrutura com pares chave-valor.  
    - *Exemplos*: metadados de imagem, configurações de usuário, resposta de API.
  - **Registro aninhado**: Estrutura com múltiplos níveis de informação.  
    - *Exemplos*: histórico de eventos, prontuário com múltiplas medições, sequência de interações.

> **NOTA**: o tipo físico de uma variável nem sempre corresponde ao seu significado estatístico. Um `id_cliente = 1532`, por exemplo, é armazenado como número, mas não deve ser tratado como variável numérica quantitativa pois este número é arbitrariamente distribuído aos clientes e não representa nenhum tipo de grandeza real.

---

# Formato de Dados

## Estruturas de Dados Retangulares
Este é o termo comum utilizado para descrever dados distribuídos em forma de matrizes bidimensionais, possuindo colunas indicando registros e colunas indicando características (features/variáveis).

Em linguagens como R e Python nos referimos a esta classe de dados como *data frames*.

Muitos dados possuem natureza não estruturada (e.g., texto/imagem/video). Dados contidos em bancos de dados relacionais precisam ser extraídos e organizados em uma tabela para a maior parte das tarefas de análise de dados e modelagem.

- **Data frame**: Dados retangulares (como tabelas) são a estrutura básica da estatística e dos modelos de *machine learning*.
- **Feature**: Uma coluna contida em uma tabela é comumnete chamada de *feature*/característica.
  - *Sinônimos*: atributo, input/entrada, preditores, variáveis.
- **Outcome**: É aquilo que se espera de saída de um modelo, também se enquadram nas classes de dados anteriormente apresentadas e definem o tipo de atividade e modelo utilizado como classificação, regressão, etc.
  - *Sinônimos*: variável dependente, resposta, objetivo/*target*, output/saída.
- **Registros**: Uma linha dentro de uma tabela é comumente referida como um registro.
  - *Sinônimos*: caso, exemplo, instância, observação, padrão, amostra.

Existem certos padrões de dados que podemos identificar para tirar máximo proveito destas estruturas matriciais que se correlacionam diretamente com os *outcomes* que desejamos dos nossos modelos.

> **NOTA**: Tabelas que possuem variáveis categóricas binárias presentes na sua coluna mais a direita geralmente são montadas pensando em tarefas de classificação, onde as colunas de $0$ à $n-1$ são utilizadas como *features* enquanto a coluna $n$ é utilizada como dado anotado para cálculo de erro ajudando os modelos a convergirem.

> **ATENÇÃO**: As áreas da estatística, ciência de dados e ciência da computação utilizam diferentes termos quando se referem aos mesmos objetos. Sejam criteriosos com as nomenclaturas utilizadas e consultem a literatura disponível nestas áreas afim de sanar confusões. Também é importante padronizar o trabalho, procurem homogeneidade no uso dos termos quando produzirem qualquer tipo de material, código, relatório.

## Estruturas de Dados Não-Retangulares

> **NOTA**: Formas de dados não retangulares fogem ao escopo do curso e não serão abordadas em profundidade. Cada uma destas formas exige metodologias alternativas específicas para processamento e análise dos dados.

Algumas destas estruturas alternativas são:

- **Séries temporais**: Registros sucessivos mensurando uma ou mais variáveis ao longo do tempo. São a base de muitos métodos estatísticos preditivos, como previsão de demanda, análise de tendência e detecção de sazonalidade.

- **Estruturas de dados espaciais**: Dados associados a localização, distância, vizinhança ou geometria. Podem ser representados por pontos, linhas, polígonos, grades, mapas ou coordenadas geográficas.

> Em muitos projetos práticos, dados não-retangulares são transformados em representações retangulares por meio de extração de atributos, agregações, vetorização ou engenharia de features.

# Dataset Titanic
O dataset Titanic é útil porque é pequeno, intuitivo, historicamente situado e cheio de problemas comuns em dados.

**Interpretação**: 

- Cada linha representa um passageiro.
- Cada coluna representa um atributo registrado sobre esse passageiro.
- A variável-alvo é a sobrevivência.
- A pergunta de modelagem é:

Podemos usar informações dos passageiros para estimar a probabilidade de sobrevivência?

## Estrutura
| Variável   | Significado                | Tipo estatístico   |
|------------|----------------------------|--------------------|
| `survived` | Se o passageiro sobreviveu | Categórica binária |
| `pclass`   | Classe do bilhete          | Categórica ordinal |
| `sex`      | Sexo registrado            | Categórica nominal |
| `age`      | Idade do passageiro        | Numérica contínua  |
| `sibsp`    | Irmãos/cônjuges a bordo    | Numérica discreta  |
| `parch`    | Pais/filhos a bordo        | Numérica discreta  |
| `fare`     | Tarifa paga                | Numérica contínua  |
| `embarked` | Porto de embarque          | Categórica nominal |
| `ticket`   | Número do bilhete          | Categórica nominal |
| `cabin`    | Número da cabine           | Categórica nominal |

> Importante notar que os termos "categórica" e "numérica" para tipo estatístico são equivalentes à "qualitativa" e "quantitativa", respectivamente. A preferência aqui se dá pelo contexto de nossa matéŕia. Materiais com enfoque em estatística optarão pela segunda opção enquanto, na área de machine learing, linguagens como Python/R e suas bibliotecas dão preferência à primeira forma.

# Referências
- [Quantiles and Percentiles, Clearly Explained!!!](https://youtu.be/IFKQLDmRK0Y?si=vLNkYOdekx3-S3qK)
- [Boxplots are Awesome!!!](https://youtu.be/fHLhBnmwUM0?si=oItzrsp146SKh3Xi)
- [How to read a box plot (a.k.a. a box-and-whisker plot) - Nick Desbarats](https://youtu.be/iBq23-eQhp8?si=_wC8MLx5p3Qk1X_y)

# Bibliografia
- Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python
