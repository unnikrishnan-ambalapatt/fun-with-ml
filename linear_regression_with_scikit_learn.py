import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn import linear_model
import pickle

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
X = np.array(data.drop(labels=[predict], axis=1))
y = np.array(data[predict])

best_accuracy = 0
for _ in range(30):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear_modl = linear_model.LinearRegression()
    linear_modl.fit(X_train, y_train)
    accuracy = linear_modl.score(X_test, y_test)
    print('Accuracy: ' + str(accuracy))
    print('Coefficient: ' + str(linear_modl.coef_))
    print('Intercept: ' + str(linear_modl.intercept_))

    predictions = linear_modl.predict(X_test)
    for x in range(len(predictions)):
        print(predictions[x], X_test[x], y_test[x])

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print('Best accuracy: ' + str(best_accuracy))
        with open('studentmodel.pickle', 'wb') as f:
            pickle.dump(linear_modl, f)

pickle_input = open('studentmodel.pickle', 'rb')
model_from_pickle = pickle.load(pickle_input)
predictions_from_pickle = model_from_pickle.predict(X_test)
for x in range(len(predictions_from_pickle)):
    print(predictions_from_pickle[x], X_test[x], y_test[x])