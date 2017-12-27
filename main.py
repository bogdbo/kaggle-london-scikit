import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize

train = pd.read_csv('./train.csv', header=None).values.astype(float)
test = pd.read_csv('./test.csv', header=None).values.astype(float)
labels = pd.read_csv('./trainLabels.csv', header=None).values.astype(float).ravel()

pca = PCA(n_components=0.9, svd_solver='full') # 40K -> 34K with 98.3% variance retained
trainr = pca.fit_transform(train)
print(trainr.shape)
variance = pca.explained_variance_ratio_.cumsum()

randIndices = np.random.permutation(train.shape[0])
split = 800;
trainIndices, testIndices = randIndices[:split], randIndices[split:]
Xtrain, Xtest = trainr[trainIndices], trainr[testIndices]
Ytrain, Ytest = labels[trainIndices], labels[testIndices]

print('SVC')
svc = SVC()
svc.fit(Xtrain, Ytrain)
print(svc.score(Xtrain, Ytrain))
print(svc.score(Xtest, Ytest))

print('RF')
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(Xtrain, Ytrain)
print(rfc.score(Xtrain, Ytrain))
print(rfc.score(Xtest, Ytest))

# res = svc.predict(test)
# column = res.reshape(-1, 1).astype(int) # unknown rows, 1 column
# df = pd.DataFrame(column)
# df.index += 1
# df.index.name = 'Id'
# df.to_csv('./result.csv', header=['Solution'])

