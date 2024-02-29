from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, model_selection

data_frame = pd.read_csv("car.data")
print(data_frame.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data_frame["buying"]))
maint = le.fit_transform(list(data_frame["maint"]))
door = le.fit_transform(list(data_frame["door"]))
persons = le.fit_transform(list(data_frame["persons"]))
leg_boot = le.fit_transform(list(data_frame["leg_boot"]))
safety = le.fit_transform(list(data_frame["safety"]))
cls = le.fit_transform(list(data_frame["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, leg_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)