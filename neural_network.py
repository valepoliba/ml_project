import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

class NeuralNetwork():

    # 3 layers with 10 node first layer...wait and bias
    def __init__(self, learning_rate=0.001, lmd=0, epochs=100, layers=[10, 5, 5]):
        self.w = {}
        self.b = {}
        self.layers = layers
        self.learning_rate = learning_rate
        self.lmd = lmd
        self.epochs = epochs
        self.X = None
        self.y = None
        self.loss = []
        # variable for the competition
        self.A = {}
        self.Z = {}
        self.dW = {}
        self.dB = {}
        self.dA = {}
        self.dZ = {}

    def init_params(self):
        L = len(self.layers)

        for l in range(1, L):
            self.w[l] = np.random.randn(self.layers[l], self.layers[l-1])
            self.b[l] = np.zeros((self.layers[l], 1))

    def sigmoid(self, Z):
        return 1/(1 + np.exp(-Z))

    def sigmoid_derivative(self, A):
        g_prime = A * (1 - A)
        return g_prime

    def forward_propagation(self):
        layers = len(self.w)
        values = {}
        for i in range(1, layers+1):
            # per ogni layer, se è il primo moltiplica l'input con il wait of the firt layer and add bias del primo livello
            # poi activate l z value con il sigmoind function
            if i == 1:
                values['Z' + str(i)] = np.dot(self.w[i], self.X.T) + self.b[i]
                values['A' + str(i)] = self.sigmoid(values['Z' + str(i)])
            # wait per ogni layer con l'activate del precedente layer
            else:
                values['Z' + str(i)] = np.dot(self.w[i], values['A'+str(i - 1)]) + self.b[i]
                values['A' + str(i)] = self.sigmoid(values['Z' + str(i)])
        return values

    def compute_cost(self, AL):
        m = self.y.shape[0]
        layers = len(AL) // 2
        # last layer is the prediction
        Y_pred = AL['A'+str(layers)]

        cost = -np.average(self.y.T * np.log(Y_pred) + (1 - self.y.T) * np.log(1 - Y_pred))
        reg_sum = 0
        for l in range(1, layers):
            reg_sum += (np.sum(np.square(self.w[l])))
        # lezione 4.3 slide 21
        L2_reg = reg_sum * (self.lmd / (2*m))

        return cost + L2_reg

    def compute_cost_derivative(self, AL):
        return -(np.divide(self.y.T, AL) - np.divide(1 - self.y.T, 1 - AL))

    def backpropagation_step(self, values):
        layers = len(self.w)
        m = self.X.shape[0]
        params_upd = {}
        # al contrario, dall'ultimo al primo layer
        for i in range(layers, 0, -1):

            if i == layers:
                dA = self.compute_cost_derivative(values['A'+str(i)])
                # local update dell'ultimo layer
                dZ = np.multiply(dA, self.sigmoid_derivative(values['A'+str(i)]))
            else:
                # wait value and previous
                dA = np.dot(self.w[i+1].T, dZ)
                dZ = np.multiply(dA, self.sigmoid_derivative(values['A'+str(i)]))
            # propagate per l'update, slide 32, propagazione per ogni layer
            if i == 1:
                params_upd['W'+str(i)] = (1 / m) * (np.dot(dZ, self.X) + self.lmd * self.w[i])
                params_upd['B'+str(i)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            else:
                params_upd['W' + str(i)] = (1 / m) * (np.dot(dZ, values['A'+ str(i - 1)].T) + self.lmd * self.w[i])
                params_upd['B' + str(i)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        return params_upd
    # adjust w and b final prediction more near the reality values
    def update(self, upd):
        layers = len(self.w)
        for i in range(1, layers+1):
            self.w[i] = self.w[i] - self.learning_rate * upd['W'+str(i)]
            self.b[i] = self.b[i] - self.learning_rate * upd['B'+str(i)]

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.init_params()

        for i in range(self.epochs):
            A_list = self.forward_propagation()
            cost = self.compute_cost(A_list)
            grads = self.backpropagation_step(A_list)
            self.update(grads)
            self.loss.append(cost)

    def predict(self, X_test):
        self.X = X_test
        AL = self.forward_propagation()
        layers = len(AL) // 2
        Y_preds = AL['A' + str(layers)]
        return np.round(Y_preds)

    def accuracy(self, y, y_preds):
        acc = float(sum(y == y_preds) / len(y) * 100)
        return acc

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss curve")
        plt.show()