import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, iN, oN, epochs=50):
        self.N = iN
        self.M = oN
        self.epochs = epochs

    def __call__(self, plots, inputs, outputs):
        x, y = np.array(inputs), np.array(outputs)
        bias = -1
        for epoch in range(self.epochs):
            for i in self.axis:
                if i == self.axis[0]:
                    self.layers[i] = x @ self.weights[i] + bias
                    self.slayers[i] = self.sigmoid(self.layers[i])
                else:
                    self.layers[i] = self.layers[i+1] @ self.weights[i] + bias
                    self.slayers[i] = self.sigmoid(self.layers[i])

            for i in self.raxis:
                if i == self.raxis[0]:
                    error = (y - self.slayers[i])**2
                    delta = error*self.sigmoid(self.layers[i], derv=True)
                else:
                    error = self.weights[i-1] @ delta
                    delta = error * self.sigmoid(self.layers[i], derv=True)

                self.weights[i] -= 3nn.0*delta

        for j, i in enumerate(self.axis):
            z = self.weights[i]
            mm, nn = z.shape
            xx, yy = np.meshgrid(range(nn), range(mm))
            plots[j].cla()
            plots[j].set_title(f'Weights: {j + 1}')
            plots[j].contourf(xx, yy, z, cmap='jet')

        plt.pause(0.001)

    def sigmoid(self, x, derv=False):
        f = 1.0/(1.0 + np.exp(-x))
        if derv:
            return f*(1 - f)
        return f

    def buildParams(self):
        N, M = self.N, self.M
        
        self.axis = list(range(N, M, -1))
        self.raxis = self.axis[::-1]

        self.weights = {}
        self.layers = {}
        self.slayers = {}

        for i in self.axis:
            self.weights[i] = np.random.rand(i, i-1)
            self.layers[i] = np.zeros(i - 1)
            self.slayers[i] = np.zeros(i - 1)


N = 300
X = np.random.rand(N, 6)
Y = np.random.rand(N, 2)

net = NeuralNetwork(6, 2)
net.buildParams()

fig = plt.figure(figsize=(9, 8))
plots = [fig.add_subplot(u) for u in (221,222,223,224)]

for i, (ix, iy) in enumerate(zip(X, Y)):
    print("rows left: ", N - i)
    net(plots, ix, iy)

plt.show()
