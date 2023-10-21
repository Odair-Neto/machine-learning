from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import preprocessing
import matplotlib.pyplot as plt
from random import shuffle
from numpy import ravel

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
    gravidade_training.append([sample.gravidade])
    gravidade_training_clf.append([sample.gravidade])
    classe_grav_training.append([sample.classe_grav])

scaler_in = preprocessing.StandardScaler().fit(input_data_training)
scaler_grav = preprocessing.StandardScaler().fit(gravidade_training)
scaler_class_grav = preprocessing.StandardScaler().fit(classe_grav_training)
print(scaler_in.mean_)
X_scaled = scaler_in.transform(input_data_training)
G_scaled = scaler_grav.transform(gravidade_training)

# Parameters
reg = MLPRegressor(solver="lbfgs",hidden_layer_sizes=(20,5,3),activation="logistic",
                   learning_rate_init=0.01,max_iter=1500, 
                   random_state=42)
reg.fit(X_scaled, G_scaled)
print("============TRAINING============")
print("Regression MLP score:")
print(reg.score(X_scaled, G_scaled))
# print("Regression MLP layers:")
# print(reg.n_layers_)
# print("Regression MLP outputs:")
# print(reg.n_outputs_)
# print("Regression MLP bias vectors:")
# print(reg.intercepts_)
# print("Regression MLP features:")
# print(reg.n_features_in_)
# print("Regression MLP weights:")
# print(reg.coefs_)
clf = MLPClassifier(hidden_layer_sizes=(10,10,),activation="logistic",
                    learning_rate_init=0.01,
                    max_iter=2000, random_state=42).fit(X_scaled, classe_grav_training)
print("Classifier MLP score:")
# gravidade_estimada = reg.predict(X_scaled)
# gravidade_buffer = []
# for grav in gravidade_estimada:
#     gravidade_buffer.append([grav])
print(clf.score(X_scaled, classe_grav_training))
# print("Classification MLP weights:")
# print(clf.coefs_)
#print(clf.score(gravidade_2, classe_grav))

#============TESTING=================================================================
input_data_testing = []
gravidade_testing = []
classe_grav_testing = []
for sample in samples_testing:
    input_data_testing.append([sample.qpa, sample.pulso, sample.resp])
    gravidade_testing.append([sample.gravidade])
    classe_grav_testing.append([sample.classe_grav])

X_test_scaled = scaler_in.transform(input_data_testing)
G_test_scaled = scaler_grav.transform(gravidade_testing)
print("")
print("============TESTING============")
print("Regression MLP score:")
print(reg.score(X_test_scaled, G_test_scaled))
print("Classifier MLP score:")
# gravidade_estimada = reg.predict(input_data_testing)
# gravidade_buffer = []
# for grav in gravidade_estimada:
#     gravidade_buffer.append([grav])
print(clf.score(X_test_scaled, classe_grav_testing))

#============PLOTTING=================================================================
# plt.figure()
# plot_tree(reg, filled=True, fontsize=6)
# plt.title("Árvore de Regressão - Gravidade")

# plt.figure()
# plot_tree(clf, filled=True, fontsize=7)
# plt.title("Árvore de Decisão - Classe de Gravidade")
# plt.show()