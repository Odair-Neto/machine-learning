import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from random import shuffle

class Sample():
    def __init__(self, id, qpa, pulso, resp, gravidade, classe_grav):
        self.id = id
        self.qpa = qpa
        self.pulso = pulso
        self.resp = resp
        self.gravidade = gravidade
        self.classe_grav = classe_grav

samples_training = []
samples_testing = []
arq = open("../treino_sinais_vitais_com_label.txt")
linhas = arq.readlines()
shuffle(linhas)
n_linha=0
for linha in linhas:
    n_linha+=1
    sample = linha.split(',')
    if n_linha > 0.7*len(linhas):
        samples_training.append(Sample(int(sample[0]), float(sample[3]), float(sample[4]), 
                          float(sample[5]), float(sample[6]), int(sample[7])))
    else:
        samples_testing.append(Sample(int(sample[0]), float(sample[3]), float(sample[4]), 
                          float(sample[5]), float(sample[6]), int(sample[7])))

#============TRAINING=================================================================
input_data_training = []
gravidade_training = []
gravidade_training_clf = []
classe_grav_training = []
for sample in samples_training:
    input_data_training.append([sample.qpa, sample.pulso, sample.resp])
    gravidade_training.append(sample.gravidade)
    gravidade_training_clf.append([sample.gravidade])
    classe_grav_training.append(sample.classe_grav)

# Parameters
reg = DecisionTreeRegressor(min_samples_split=50).fit(input_data_training, gravidade_training)
print("============TRAINING============")
print("Regression tree score:")
print(reg.score(input_data_training, gravidade_training))
clf = DecisionTreeClassifier(criterion="entropy",
                              min_samples_split=10).fit(gravidade_training_clf, classe_grav_training)
print("Classifier tree score:")
gravidade_estimada = reg.predict(input_data_training)
gravidade_buffer = []
for grav in gravidade_estimada:
    gravidade_buffer.append([grav])
print(clf.score(gravidade_buffer, classe_grav_training))
#print(clf.score(gravidade_2, classe_grav))

#============TESTING=================================================================
input_data_testing = []
gravidade_testing = []
classe_grav_testing = []
for sample in samples_testing:
    input_data_testing.append([sample.qpa, sample.pulso, sample.resp])
    gravidade_testing.append(sample.gravidade)
    classe_grav_testing.append(sample.classe_grav)

print("")
print("============TESTING============")
print("Regression tree score:")
print(reg.score(input_data_testing, gravidade_testing))
print("Classifier tree score:")
gravidade_estimada = reg.predict(input_data_testing)
gravidade_buffer = []
for grav in gravidade_estimada:
    gravidade_buffer.append([grav])
print(clf.score(gravidade_buffer, classe_grav_testing))

#============PLOTTING=================================================================
plt.figure()
plot_tree(reg, filled=True, fontsize=6)
plt.title("Árvore de Regressão - Gravidade")

plt.figure()
plot_tree(clf, filled=True, fontsize=7)
plt.title("Árvore de Decisão - Classe de Gravidade")
plt.show()