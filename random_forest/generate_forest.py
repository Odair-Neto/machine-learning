from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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
reg = RandomForestRegressor(min_samples_split=20, n_estimators= 100, 
                            max_features=3).fit(input_data, gravidade)
print("Regression forest score:")
print(reg.score(input_data, gravidade))
clf = RandomForestClassifier(criterion="entropy", min_samples_split=20,
                              n_estimators=100, max_features=2).fit(gravidade_2, classe_grav)
print("Classifier forest score:")
#print(clf.score(gravidade_2, classe_grav))
gravidade_estimada = reg.predict(input_data)
gravidade_estimada_2 = []
for grav in gravidade_estimada:
    gravidade_estimada_2.append([grav])
print(clf.score(gravidade_estimada_2, classe_grav))
#scores = cross_val_score(reg,input_data,gravidade, cv=5)
#print(scores.mean())

plot_step = 0.02  # fine step width for decision surface contours
plot_step_coarser = 0.5  # step widths for coarse classifier guesses
cmap = plt.cm.RdYlBu
RANDOM_SEED = 13  # fix the seed on each iteration

plt.subplot(3, 4, 1)
    # Add a title at the top of each column
plt.title("Random Forest", fontsize=9)
# Now plot the decision boundary using a fine mesh as input to a
# filled contour plot
input_data_2 = []
input_data_3 = []
for data in input_data:
    input_data_2.append(data[1])
    input_data_3.append(data[2])
plt.scatter(
    input_data_2,
    input_data_3,
    c=classe_grav,
    cmap=ListedColormap(["r", "y", "b"]),
    edgecolor="k",
    s=20,
)

plt.suptitle("Classe de gravidade", fontsize=12)
plt.axis("tight")
plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
plt.show()

# plt.figure()
# plot_tree(reg, filled=True, fontsize=6)
# plt.title("Árvore de Regressão - Gravidade")

# plt.figure()
# plot_tree(clf, filled=True, fontsize=7)
# plt.title("Árvore de Decisão - Classe de Gravidade")
# plt.show()