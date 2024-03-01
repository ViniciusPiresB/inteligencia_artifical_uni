from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

# 0 - cachorro
# 1 - porco

treino_y = [1, 1, 1, 0, 0, 0]

modelo = LinearSVC()

modelo.fit(treino_x, treino_y)

a = [1, 1, 1]

result = modelo.predict([a])

if result == 0:
    print("cachorro")
else:
    print("porco")

am1 = [1, 1, 1]
am2 = [1, 1, 0]
am3 = [0, 1, 1]

teste_x = [am1, am2, am3]
teste_y = [0, 1, 1]

previsoes = modelo.predict(teste_x)

for prev in previsoes:
    if(prev == 0):
        print("cachorro ")
        continue

    print("porco ")

print(previsoes)

acuracia = accuracy_score(teste_y, previsoes)

print(acuracia)

salva_dummy = []
testes = 300

summ = 0

for i in range(testes):
    dummy = DummyClassifier(strategy="uniform")
    dummy.fit(teste_x, teste_y)

    pred_dummy = dummy.predict(teste_x)
    acuracia_dummy = accuracy_score(teste_y, pred_dummy)
    summ += acuracia_dummy
    print(acuracia_dummy)

print(f"media: {summ/300}")