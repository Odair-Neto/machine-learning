from sklearn.datasets import load_iris

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree

class Sample():
    def __init__(self, id, qpa, pulso, resp, gravidade, classe_grav):
        self.id = id
        self.qpa = qpa
        self.pulso = pulso
        self.resp = resp
        self.gravidade = gravidade
        self.classe_grav = classe_grav

samples = []
arq = open("../treino_sinais_vitais_com_label.txt")
linhas = arq.readlines()
for linha in linhas:
    sample = linha.split(',')
    samples.append(Sample(int(sample[0]), float(sample[3]), float(sample[4]), 
                          float(sample[5]), float(sample[6]), int(sample[7])))

input_data = []
gravidade = []
gravidade_2 =[]
classe_grav = []
for sample in samples:
    input_data.append([sample.qpa, sample.pulso, sample.resp])
    gravidade.append(sample.gravidade)
    gravidade_2.append([sample.gravidade])
    classe_grav.append(sample.classe_grav)

# Parameters
reg = DecisionTreeRegressor(min_samples_split=20).fit(input_data, gravidade)
print("Regression tree score:")
print(reg.score(input_data, gravidade))
clf = DecisionTreeClassifier(criterion="entropy", min_samples_split=20).fit(gravidade_2, classe_grav)
print("Classifier tree score:")
#print(clf.score(input_data, classe_grav))
reg.score()
plt.figure()
plot_tree(reg, filled=True, fontsize=4)
plt.title("Árvore de Regressão - Gravidade")

plt.figure()
plot_tree(clf, filled=True, fontsize=4)
plt.title("Árvore de Decisão - Classe de Gravidade")
plt.show()