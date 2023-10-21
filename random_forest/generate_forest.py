import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from random import shuffle
from sklearn.tree import plot_tree

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

#========================TRAINING=============================
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
reg = RandomForestRegressor(min_samples_split=5, n_estimators= 100, 
                            max_features=2).fit(input_data_training, gravidade_training)
print("========TRAINING========")
print("Regression forest score:")
print(reg.score(input_data_training, gravidade_training))
clf = RandomForestClassifier(criterion="entropy", min_samples_split=20, n_estimators=10, 
                             max_features=3).fit(gravidade_training_clf, classe_grav_training)
print("Classifier forest score:")
#print(clf.score(gravidade_2, classe_grav))
gravidade_estimada = reg.predict(input_data_training)
gravidade_estimada_2 = []
for grav in gravidade_estimada:
    gravidade_estimada_2.append([grav])
print(clf.score(gravidade_estimada_2, classe_grav_training))


#============TESTING=================================================================
input_data_testing = []
gravidade_testing = []
classe_grav_testing = []
for sample in samples_testing:
    input_data_testing.append([sample.qpa, sample.pulso, sample.resp])
    gravidade_testing.append(sample.gravidade)
    classe_grav_testing.append(sample.classe_grav)

print("")
print("========TESTING========")
print("Regression tree score:")
print(reg.score(input_data_testing, gravidade_testing))
print("Classifier tree score:")
gravidade_estimada = reg.predict(input_data_testing)
gravidade_buffer = []
for grav in gravidade_estimada:
    gravidade_buffer.append([grav])
print(clf.score(gravidade_buffer, classe_grav_testing))

# #Plotar gravidade x classe de gravidade
# plt.figure()
# plt.title("Gravidade x Classe Gravidade", fontsize=9)
# plt.scatter(
#     gravidade_training,
#     classe_grav_training,
#     edgecolor="k",
#     s=20,
# )
# plt.show()



# plot_tree(clf., filled=True, fontsize=7)
# plt.title("Árvore de Decisão - Classe de Gravidade")
# plt.show()

#scores = cross_val_score(reg,input_data,gravidade, cv=5)
#print(scores.mean())


#     # Add a title at the top of each column

# Now plot the decision boundary using a fine mesh as input to a
# filled contour plot
# input_data_2 = []
# input_data_3 = []
# input_data_4 = []
# input_data_5 = []
# for data in input_data:
#     input_data_2.append(data[0])
#     input_data_3.append(data[1])
#     input_data_4.append(data[2])
#     input_data_5.append([data[0], data[1]])


# ax.set_xlabel("qPA")
# ax.set_ylabel("Pulso (bpm)")

# ax = plt.subplot(3, 1, 2)
# plt.legend()
# plt.scatter(
#     input_data_2,
#     input_data_4,
#     c=classe_grav,
#     cmap=ListedColormap(["g", "y", "orange", "r"]),
#     edgecolor="k",
#     s=20,
# )
# ax.set_xlabel("qPA")
# ax.set_ylabel("FpM")

# ax = plt.subplot(3, 1, 3)
# plt.scatter(
#     input_data_3,
#     input_data_4,
#     c=classe_grav,
#     cmap=ListedColormap(["r", "orange", "y", "g"]),
#     edgecolor="k",
#     s=20,
# )
# ax.set_xlabel("Pulso (bpm)")
# ax.set_ylabel("FpM")

# plt.suptitle("Classe de gravidade", fontsize=12)
# plt.axis("tight")
# plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)


# plt.figure()
# plot_tree(reg, filled=True, fontsize=6)
# plt.title("Árvore de Regressão - Gravidade")

# plt.figure()
# plot_tree(clf, filled=True, fontsize=7)
# plt.title("Árvore de Decisão - Classe de Gravidade")
# plt.show()