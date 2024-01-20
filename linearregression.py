import numpy as np
import matplotlib.pyplot as plt


def linear_regression_model(x, y, learning_rate, iteration):
    m = y.size
    theta = np.zeros((2, 1))
    for i in range(iteration):
        y_prediction = np.dot(x, theta)
        d_theta = (1/m)*np.dot(x.T, y_prediction - y)
        theta = theta - learning_rate*d_theta
    return theta


data = np.loadtxt('data.txt', delimiter=',')
X = data[:, 0]
Y = data[:, 1].reshape(X.size, 1)
X = np.vstack((np.ones((X.size, )), X)).T
plt.scatter(X[:, 1], Y)
plt.show()
iteration_count = 1000
learn_rate = 0.00000005
theta_result = linear_regression_model(X, Y, learning_rate=learn_rate, iteration=iteration_count)
new_houses = np.array([[1, 1547], [1, 1896], [1, 1934], [1, 2800], [1, 3400], [1, 5000]])
for house in new_houses:
    print("Predicted price of house with", house[1], "sq. ft. area is: $",
          round(np.dot(house, theta_result)[0], 2))
