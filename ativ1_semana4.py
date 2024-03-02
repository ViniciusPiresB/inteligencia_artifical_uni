from sklearn.svm import LinearSVC
import numpy as np

# [usa roupa, fala, se move, esta invisivel, tem maos, nao tem face]
humano1 = [1, 1, 1, 0, 1, 0]
humano2 = [0, 1, 1, 0, 1, 1]
humano3 = [1, 0, 1, 0, 1, 0]
humano4 = [1, 1, 1, 0, 1, 0]
humano5 = [1, 0, 1, 0, 0, 1]
humano6 = [1, 1, 1, 0, 1, 0]
humano7 = [1, 0, 0, 0, 1, 0]
humano8 = [0, 1, 1, 0, 0, 0]
humano9 = [0, 1, 0, 0, 1, 0]
humano10 = [0, 1, 1, 0, 1, 0]
humano11 = [1, 1, 0, 0, 1, 0]
humano12 = [1, 1, 1, 0, 1, 0]
humano13 = [1, 1, 1, 0, 0, 0]
humano14 = [1, 1, 0, 0, 1, 1]
humano15 = [1, 1, 1, 0, 1, 0]
humano16 = [0, 1, 1, 0, 1, 0]
humano17 = [1, 1, 1, 0, 1, 0]
humano18 = [1, 1, 0, 0, 1, 0]
humano19 = [1, 0, 1, 0, 1, 0]
humano20 = [1, 1, 0, 0, 0, 0]

# [usa roupa, fala, se move, esta invisivel, tem maos, nao tem face]
espirito1 = [0, 0, 1, 0, 1, 1]
espirito2 = [0, 1, 0, 1, 0, 1]
espirito3 = [0, 0, 1, 1, 1, 1]
espirito4 = [0, 0, 0, 1, 1, 0]
espirito5 = [0, 0, 1, 0, 1, 0]
espirito6 = [0, 0, 0, 1, 1, 1]
espirito7 = [0, 1, 1, 0, 0, 1]
espirito8 = [1, 0, 0, 1, 0, 1]
espirito9 = [0, 0, 1, 1, 1, 0]
espirito10 = [0, 0, 0, 1, 0, 1]
espirito11 = [0, 0, 1, 1, 0, 0]
espirito12 = [0, 1, 0, 1, 0, 1]
espirito13 = [0, 0, 1, 0, 1, 1]
espirito14 = [1, 0, 0, 1, 0, 0]
espirito15 = [0, 0, 1, 1, 1, 1]
espirito16 = [0, 0, 0, 1, 0, 0]
espirito17 = [0, 0, 1, 1, 0, 1]
espirito18 = [0, 1, 0, 1, 1, 1]
espirito19 = [0, 0, 1, 0, 0, 1]
espirito20 = [0, 1, 0, 1, 1, 0]

treino_x = np.array([humano1, humano2, humano3, humano4, humano5, humano6, humano7, humano8, humano9, humano10, humano11, humano12, humano13, humano14, humano15, humano16, humano17, humano18, humano19, humano20, espirito1, espirito2, espirito3, espirito4, espirito5, espirito6, espirito7, espirito8, espirito9, espirito10, espirito11, espirito12, espirito13, espirito14, espirito15, espirito16, espirito17, espirito18, espirito19, espirito20])

# 1 - pessoa
# 0 - fantasma

resultado_esperado = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

modelo = LinearSVC()

modelo.fit(treino_x, resultado_esperado)

pessoaPredict = np.array([1, 1, 1, 0, 1, 0]).reshape(1, -1)
fantasmaPredict = np.array([0, 0, 1, 1, 0, 1]).reshape(1, -1)

pessoaResult = modelo.predict(pessoaPredict)
fantasmaResult = modelo.predict(fantasmaPredict)

for result in [pessoaResult, fantasmaResult]:
    if result == 0:
        print("fantasma")
    else:
        print("pessoa")

print(len(resultado_esperado))