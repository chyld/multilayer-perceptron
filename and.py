import numpy as np


def initialize():
    w = np.random.random(2)
    b = np.random.random()
    return w, b


def feedforward(w, x, b):
    return activation(w @ x + b)


def activation(z):
    return 1 if z > 0 else 0


def backprop(w, x, b, y, alpha):
    yhat = feedforward(w, x, b)
    dw1 = -alpha * 2 * (yhat - y) * x[0]
    dw2 = -alpha * 2 * (yhat - y) * x[1]
    db = -alpha * 2 * (yhat - y) * 1
    return dw1, dw2, db


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])
alpha = 0.1
w, b = initialize()
print('w:', w, 'b:', b)
for _ in range(100):
    for i, x in enumerate(X):
        dw1, dw2, db = backprop(w, x, b, y[i], alpha)
        w[0] += dw1
        w[1] += dw2
        b += db
        print('--> w:', w, 'b:', b)

print('w:', w, 'b:', b)

print('0 0:', feedforward([0, 0], w, b))
print('0 1:', feedforward([0, 1], w, b))
print('1 0:', feedforward([1, 0], w, b))
print('1 1:', feedforward([1, 1], w, b))
