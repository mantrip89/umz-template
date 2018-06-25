
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
dataframe = pandas.read_csv("train/train.tsv", sep='\t', header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = pandas.DataFrame(dataframe, columns=[2,3,5])
Y = dataset[:,0]

# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 7
#numpy.random.seed(seed)
# evaluate model with standardized dataset
#estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(estimator, X, Y, cv=kfold)
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# evaluate model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
pipeline.fit(X, Y)
prediction = pipeline.predict(X)
#print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

dataframe = pandas.read_csv("dev-0/in.tsv", sep='\t', header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X_dev = pandas.DataFrame(dataframe, columns=[1,2,4])
Y_dev = pandas.read_csv('dev-0/expected.tsv',
                    header=None, names=["price"], sep='\t')

predictions_dev = pipeline.predict(X_dev)

with open('dev-0/out.tsv', 'w') as file:
    for prediction in predictions_dev:
        file.write(str(prediction) + '\n')

dataframe = pandas.read_csv("test-A/in.tsv", sep='\t', header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X_test = pandas.DataFrame(dataframe, columns=[1,2,4])


predictions_test = pipeline.predict(X_test)

with open('test-A/out.tsv', 'w') as file:
    for prediction in predictions_test:
        file.write(str(prediction) + '\n')
