import pandas as pd
from matplotlib import pyplot
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import Perceptron

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["studytime", "health", "G3"]]
X = data[["studytime", "health"]]
y = data["G3"]
# print(X)
print(X.values)
p = Perceptron()
p.fit(X, y)
print(p.coef_)
print(p.intercept_)
plot_decision_regions(X.values, y.values, clf=p, legend=2)
pyplot.show()