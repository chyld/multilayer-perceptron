import numpy as np

# function definitions
# ----------------------------------------------------------------------------------------------- #


def feedforward(sample):
    S0 = W0 @ sample + B0
    A0 = np.tanh(S0)
    A1 = W1.T @ A0 + B1
    return S0, A0, A1[0][0]


# initialization phase
# ----------------------------------------------------------------------------------------------- #
W0 = np.random.random((3, 2))
B0 = np.zeros((3, 1))
W1 = np.random.random((3, 1))
B1 = np.zeros((1, 1))

X = np.array([
    [[0], [0]],
    [[0], [1]],
    [[1], [0]],
    [[1], [1]]
])

Y = np.array([0, 1, 1, 0])

alpha = 0.01

# training phase
# ----------------------------------------------------------------------------------------------- #
for epoch in range(10001):
    # initialize gradient matrices
    dW0 = np.zeros((3, 2))
    dB0 = np.zeros((3, 1))
    dW1 = np.zeros((3, 1))
    dB1 = np.zeros((1, 1))

    # mini batch gradient descent, i.e., 4 samples per batch
    for sample in range(X.shape[0]):
        # feed forward
        S0, A0, A1 = feedforward(X[sample])
        y = Y[sample]
        # backpropagation B1/W1
        dB1 += 2 * (A1 - y) * 1
        dW1 += 2 * (A1 - y) * A0
        # backpropagation B0/W0
        dB0 += 2 * (A1 - y) * W1 * (1 - (np.tanh(S0) ** 2)) * 1
        dW0 += 2 * (A1 - y) * W1 * (1 - (np.tanh(S0) ** 2)) * sample

    W0 -= alpha * dW0
    B0 -= alpha * dB0
    W1 -= alpha * dW1
    B1 -= alpha * dB1

# prediction phase
# ----------------------------------------------------------------------------------------------- #
for sample in range(X.shape[0]):
    S0, A0, A1 = feedforward(X[sample])
    y = Y[sample]
    print('FINAL-------------------------------------------------FINAL')
    print('actual:', y, 'A1:', A1)
