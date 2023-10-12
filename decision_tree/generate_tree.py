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
output_data = []
for sample in samples:
    input_data.append([sample.qpa, sample.pulso, sample.resp])
    output_data.append(sample.gravidade)

# Parameters
plt.figure()
clf = DecisionTreeRegressor(min_samples_split=20).fit(input_data, output_data)
print(clf.score(input_data, output_data))
plot_tree(clf, filled=True, fontsize=4)
plt.title("Arvore de decisao - gravidade")
plt.show()
