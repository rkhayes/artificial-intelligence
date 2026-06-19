import numpy as np

# O objeto central no NumPy é o ndarray (que significa "n-dimensional array")
# Podemos criá-los:
# - a partir de listas Python: `np.array()` cria um array/tensor 1D
# - com funções de preenchimento: `np.zeros((3, 4))` cria uma matriz 3x4 preenchida com zeros
# - com sequências numéricas: `np.arange(0, 10, 2)` cria um array contendo uma
#   sequência de valores uniformemente espaçados (ex: 0, 2, 4, 6, 8). O np.linspace(0, 10, 5)
#   cria um número exato de valores linearmente espaçados em um determinado intervalo.
arr = np.zeros((3, 4))

print(arr)
print(np.arange(0, 10, 2))
print(np.linspace(0, 10, 5))

# Retorna o formato do array no formato (linhas, colunas)
print(arr.shape)

# Retorna o número de eixos (dimensões)
print(arr.ndim)

# Retorna a quantidade total de valores
print(arr.size)

# Retorna o array com suas dimensões reorganizadas.
# Note que ele deve ser capaz de comportar o mesmo número de valores do array original.
# Além disso, se você souber de quantas linhas precisa, mas não quantas colunas (ou vice-versa),
# você pode passar -1 como argumento para a função.
print(arr.reshape((6, -1)))

# Operações aritméticas
# Os operadores aritméticos básicos +, -, *, / realizam operações elemento a elemento (element-wise)
# automaticamente ao lidar com arrays, gerando novos arrays como resultado.
arrx = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arry = np.array([[121, 232, 381], [403, 543, 681], [702, 884, 934]])

print(arrx + arry)
print(arrx - arry)
print(arrx * arry)
print(arrx / arry)
print(np.dot(arrx, arry))  # Produto escalar / Multiplicação de matrizes

# Agregações matemáticas
# Se o parâmetro axis=0, a operação é realizada nas colunas
# Se axis=1, a operação é realizada nas linhas
print(arrx.sum())
print(arrx.min())
print(arrx.max())
print(arrx.mean())

# Indexação e fatiamento (slicing)
# Podemos usar dois pontos para separar a notação padrão de fatiamento do Python em dimensões
# ao lidar com arrays NumPy.
print(arrx[1:, 1:])

# Indexação booleana
# Podemos aplicar máscaras condicionais.
# Retorna todos os valores acima de 5 extraídos do array original em um array plano (1D).
print(arrx[arrx > 5])

# Retorna todos os índices que satisfazem a condição.
print(np.where(arrx > 5))
# Também podemos usar a função na forma `np.where(arrx > 5, 1, 0)` para substituir os
# valores do array com base no fato de satisfazerem ou não a condição.

# Algumas particularidades do NumPy que você deve conhecer:
# - Ao atribuir fatias de um array a outra variável, ele não copia os dados, apenas
# cria uma visualização (view) na memória;
# - Sempre use b = a.copy() quando precisar de dados independentes;
# - Sempre valide as dimensões (shapes) das suas matrizes.

# Exercícios:
# 1. Crie um array unidimensional (1D) contendo os números de 1 a 12 usando uma função Numpy.
# 2. Transforme este array em uma matriz 2D com 3 linhas e 4 colunas.
# 3. Imprima a matriz, seu formato (shape) e a quantidade total de elementos (size).
# 4. Crie uma "máscara booleana" da matriz do primeiro exercício que identifique quais números são pares.
# 5. Use indexação booleana para criar um novo array contendo apenas os números pares extraídos da matriz.
# 6. Use a função `np.where` para criar uma nova matriz onde todos os números maiores que 6 sejam substituídos por 0.
# 7. Crie uma matriz 2x2 preenchida apenas com uns usando `np.ones`.
# 8. Crie uma segunda matriz 2x2 apenas com números pseudoaleatórios ou definidos manualmente.
# 9. Calcule a multiplicação elemento a elemento (element-wise) dessas duas matrizes.
# 10. Calcule o produto escalar (dot product) usando a função ou operador apropriado e compare os resultados.
# 11. Crie um tensor 3x3 com valores sequenciais de 1 a 9.
# 12. Calcule a soma de todos os elementos na matriz.
# 13. Encontre o maior valor de cada coluna.
# 14. Calcule a média de cada linha.
