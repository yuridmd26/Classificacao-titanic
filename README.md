# Titanic Dataset: K-Nearest Neighbors Classifier

Este projeto utiliza o conjunto de dados do Titanic para construir e avaliar um modelo de classificação usando o algoritmo K-Nearest Neighbors (k-NN). O modelo é treinado para prever a sobrevivência de passageiros com base em variáveis como classe, sexo, idade e tarifa.

## Estrutura do Código

O script está dividido nas seguintes seções:

### 1. Carregamento de Dados
O script carrega três arquivos CSV:
- `gender_submission.csv`: Um arquivo auxiliar que contém previsões genéricas de sobrevivência.
- `test.csv`: Conjunto de dados de teste.
- `train.csv`: Conjunto de dados de treino, contendo as variáveis de entrada e a variável alvo (sobrevivência).

### 2. Exploração Inicial dos Dados
São explorados os arquivos carregados, mostrando as primeiras linhas (`head()`), informações básicas (`info()`) e estatísticas descritivas (`describe()`).

### 3. Limpeza de Dados
Tratamento de valores ausentes:
- Preenchimento da coluna `Age` com a mediana.
- Preenchimento da coluna `Fare` com a mediana.
- Exclusão das colunas com valores ausentes em `Cabin` e, para o conjunto de treino, na coluna `Embarked`.

### 4. Seleção de Variáveis
As variáveis categóricas, como o sexo, são transformadas em variáveis dummy, com a coluna `Sex_male` representando a codificação binária.

### 5. Preparação dos Dados
As variáveis de entrada selecionadas são:
- `Pclass`: Classe da passagem (1ª, 2ª ou 3ª classe).
- `Sex_male`: Sexo do passageiro (1 para masculino, 0 para feminino).
- `Age`: Idade do passageiro.
- `Fare`: Valor da passagem paga.

A variável alvo (`y`) é a sobrevivência (`Survived`), que é 0 ou 1.

### 6. Treinamento e Previsão com K-Nearest Neighbors
Um modelo k-NN é instanciado com `k=3`. O conjunto de treino é dividido em treino e teste (70% para treino, 30% para teste). O modelo é treinado no conjunto de treino, e previsões são feitas no conjunto de teste.

### 7. Avaliação de Desempenho
A avaliação é feita usando as métricas:
- **Acurácia**: Proporção de previsões corretas.
- **Matriz de Confusão**: Para análise detalhada de verdadeiro positivo, verdadeiro negativo, falso positivo e falso negativo.

## Como Executar

### Pré-requisitos
Certifique-se de ter o Python e as seguintes bibliotecas instaladas:
- `pandas`
- `scikit-learn`

Para instalar as dependências, execute:
```bash
pip install pandas scikit-learn
