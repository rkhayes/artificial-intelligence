import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Carrega os dados em um DataFrame Pandas
iris = load_iris(as_frame=True)

# DataFrame completo (variáveis preditoras + variável alvo)
df = iris.frame

# Mapeia a variável alvo para os nomes reais das espécies para facilitar a leitura
df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

# Verifica a tabela de características e rótulos
print(3 * "5 PRIMEIRAS LINHAS ")
print(df.head())

# Informações sobre a estrutura dos dados
print(3 * "INFO ESTRUTURAL ")
print(df.info())

# Auxilia na detecção numérica de valores atípicos (outliers)
print(3 * "ESTATÍSTICAS DESCRITIVAS ")
print(df.describe())

# Pairplot: para visualização da relação cruzada entre todas as variáveis
# e a separabilidade das classes.
sns.pairplot(data=df.drop("target", axis=1), hue="species", diag_kind="kde")
plt.suptitle("Relação entre Variáveis (Pairplot)", y=1.02)

# Boxplot: o melhor para detectar visualmente valores atípicos em cada variável.
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.drop(["target", "species"], axis=1), orient="h", palette="Set2")
plt.title("Distribuição e detecção de valores atípicos")

plt.show()
