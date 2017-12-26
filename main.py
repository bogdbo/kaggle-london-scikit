import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

train = pd.read_csv('./train.csv', header=None).values.astype(float)
test = pd.read_csv('./test.csv', header=None).values.astype(float)
labels = pd.read_csv('./trainLabels.csv', header=None).values.astype(float).ravel()

pca = PCA(n_components=34, svd_solver='full') # 40K -> 34K with 98.3% variance retained
train_reduced = pca.fit_transform(train)
variance = pca.explained_variance_ratio_.cumsum()

# training section
randIndices = np.random.permutation(train.shape[0])
split = 800;
trainIndices, testIndices = randIndices[:split], randIndices[split:]
Xtrain, Xtest = train[trainIndices], train[testIndices]
Ytrain, Ytest = labels[trainIndices], labels[testIndices]

# svc = SVC()
# svc.fit(Xtrain, Ytrain)
# print(svc.score(Xtrain, Ytrain))
# print(svc.score(Xtest, Ytest))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, activation="relu",
                    hidden_layer_sizes=(10), random_state=1)
clf.fit(Xtrain, Ytrain)
print(clf.score(Xtrain, Ytrain))
print(clf.score(Xtest, Ytest))

# res = svc.predict(test)
# column = res.reshape(-1, 1).astype(int) # unknown rows, 1 column
# df = pd.DataFrame(column)
# df.index += 1
# df.index.name = 'Id'
# df.to_csv('./result.csv', header=['Solution'])

