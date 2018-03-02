import numpy as np


def initialize():
    w = np.random.random()
    b = np.random.random()
    return w, b


def feedforward(w, x, b):
    return (w * x) + b


def backprop(w, x, b, y, alpha):
    yhat = feedforward(w, x, b)
    dw = -alpha * 2 * (yhat - y) * x
    db = -alpha * 2 * (yhat - y) * 1
    print('x:', x, 'error:', yhat - y)
    return dw, db


x = np.arange(40)
y = np.array(list(map(lambda x: (3 * x) + 7, x)))
alpha = 0.001
w, b = initialize()
print('w:', w, 'b:', b)
for i, value in enumerate(x):
    dw, db = backprop(w, value, b, y[i], alpha)
    w += dw
    b += db

print('w:', w, 'b:', b)

for a in range(1000, 10000, 500):
    print('a:', a, 'y:', 3 * a + 7, 'yhat:', w *
          a + b, 'ratio:', 100 * (1 - ((3 * a + 7) / (w * a + b))))
