import pandas as pd
import graphviz
from sklearn.neighbors import KNeighborsClassifier


train = pd.read_csv('train.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')

train_X = pd.DataFrame(train, columns=train.columns[:-1])
train_Y = train['Class']

test_X = pd.DataFrame(test, columns=test.columns[:-1])
test_Y = test['Class']

clf = KNeighborsClassifier()
clf = clf.fit(train_X, train_Y)


print('n_neighbors=5')
print('TRAIN SET')

print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')

print(sum(clf.predict(test_X) == test_Y) / len(test_X))


clf = KNeighborsClassifier(n_neighbors=6)
clf = clf.fit(train_X, train_Y)


print('n_neighbors=6')
print('TRAIN SET')
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
print(sum(clf.predict(test_X) == test_Y) / len(test_X))


clf = KNeighborsClassifier(n_neighbors=8)
clf = clf.fit(train_X, train_Y)


print('n_neighbors=8')
print('TRAIN SET')
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
print(sum(clf.predict(test_X) == test_Y) / len(test_X))

clf = KNeighborsClassifier(n_neighbors=10)
clf = clf.fit(train_X, train_Y)


print('n_neighbors=10')
print('TRAIN SET')
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
print(sum(clf.predict(test_X) == test_Y) / len(test_X))

clf = KNeighborsClassifier(n_neighbors=2)
clf = clf.fit(train_X, train_Y)

print('n_neighbors=2')
print('TRAIN SET')
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
print(sum(clf.predict(test_X) == test_Y) / len(test_X))

clf = KNeighborsClassifier(n_neighbors=4)
clf = clf.fit(train_X, train_Y)


print('n_neighbors=4')
print('TRAIN SET')
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
print(sum(clf.predict(test_X) == test_Y) / len(test_X))

clf = KNeighborsClassifier(n_neighbors=100)
clf = clf.fit(train_X, train_Y)


print('n_neighbors=100')
print('TRAIN SET')
print(sum(clf.predict(train_X) == train_Y) / len(train_X))
print('TEST SET')
print(sum(clf.predict(test_X) == test_Y) / len(test_X))
