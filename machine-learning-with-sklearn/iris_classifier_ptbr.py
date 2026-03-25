import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

### CARREGAMENTO DOS DADOS
# Aqui, a matriz `X` contém as características (features), enquanto `y` representa os rótulos de destino (alvos).
# O argumento `as_frame=True` retorna os dados no formato DataFrame do Pandas para facilitar o manuseio.
X, y = load_iris(as_frame=True).data, load_iris(as_frame=True).target

### DIVISÃO ENTRE TREINO E TESTE
# Dividiremos o conjunto de dados entre estas quatro variáveis, com 75% para treino e 25% para teste.
#
# Obs.: Treinar e testar um modelo com os mesmos dados é um erro metodológico que leva ao
# overfitting (sobreajuste), fazendo com que o modelo falhe ao generalizar para novos dados.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=18)

### PRÉ-PROCESSAMENTO E TRANSFORMAÇÃO
# Muitos algoritmos funcionam melhor quando todos os dados estão na mesma escala. O Scikit-learn
# fornece objetos chamados "transformers" (transformadores) com os métodos `fit()` e `transform()` para isso.
#
# É fundamental que a transformação (como a padronização) aprenda os parâmetros apenas no
# conjunto de treino e aplique essa transformação tanto nos dados de treino quanto nos dados de teste reservados.
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

### SELEÇÃO DO ESTIMADOR E TREINAMENTO DO MODELO
# Os algoritmos nativos do Scikit-learn são chamados de estimadores (estimators). Alguns exemplos:
# + Random Forest (Floresta Aleatória)
# + Support Vector Machine (Máquina de Vetores de Suporte)
# + K-Nearest Neighbors (K-Vizinhos Mais Próximos)
#
# Todo estimador pode ser treinado e ajustado aos dados usando o método `fit(X, y)`.

# Instancia um modelo KNN com 5 vizinhos.
knn = KNeighborsClassifier(n_neighbors=5)

# Treina o modelo usando os dados padronizados.
knn.fit(X_train, y_train)

### PREDIÇÃO E AVALIAÇÃO
# Uma vez que o estimador está treinado, você não precisa treiná-lo novamente para prever
# os resultados de novos dados.

# Realiza a predição para a divisão de teste.
y_pred = knn.predict(X_test)

# Calcula e imprime a pontuação de acurácia.
acc = accuracy_score(y_test, y_pred)
print(f"Acurácia do Modelo: {acc}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

### EXERCÍCIOS
# 1. Explique a diferença entre o StandardScaler e pelo menos dois outros
#    escalonadores (scalers) do Scikit-learn.
# 2. Explique o que as funções `fit()` e `transform()` fazem.
