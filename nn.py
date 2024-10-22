import numpy as np
import matplotlib.pyplot as plt


class Net:

    def __init__(self, inp, otp, epochs=100):
        self.N = inp
        self.M = otp
        self.epochs = epochs

    def __call__(self, plots, inputs, outputs):
        x, y = np.array(inputs), np.array(outputs)
        bias = -1

        
        for epoch in range(self.epochs):

            for i in self.axis:
                if i == self.axis[0]:
                    self.layers[i] = inputs @ self.weights[i] + bias
                else:
                    self.layers[i] = self.layers[i+1] @ self.weights[i] + bias
                self.slayers[i] = self.sigmoid(self.layers[i])

            for i in self.raxis:
                if i == self.raxis[0]:
                    error = (y - self.slayers[i])**2
                    delta = 2.0*(y - self.slayers[i])*self.sigmoid(self.layers[i], derv=True)
                else:
                    #print(delta.shape, self.weights[i].shape, self.weights[i-1].shape)
                    error = self.weights[i-1] @ delta
                    delta = error * self.sigmoid(self.layers[i], derv=True)
                    

                self.weights[i] -= delta

            
            #print("Number of Epochs left: ", self.epochs - epoch + 1)

        for i, j in enumerate(self.axis):
            plots[i].cla()
            plots[i].set_title(f'Weights: {i+1}')

            z = self.weights[j]
            m, n = z.shape
            x, y = np.meshgrid(range(n), range(m))
            plots[i].contourf(x, y, z, cmap='jet')

        plt.pause(0.1)
            

    def sigmoid(self, x, derv=False):
        f = 1.0/(1.0 + np.exp(-x))
        if derv:
            return f*(1 - f)
        return f

    def buildWeights(self):
        N, M = self.N, self.M
        self.weights = {}
        self.layers = {}
        self.slayers = {}

        self.axis = list(range(N, M, -1))
        self.raxis = self.axis[::-1]
        
        for t in self.axis:
            self.weights[t] = np.random.rand(t, t-1)
            self.layers[t] = np.zeros(t-1)
            self.slayers[t] = np.zeros(t-1)

    def normFunction(self, x, y):
        x, y = np.array(x), np.array(y)
        return (x - np.min(x))/(np.max(x) - np.min(x)), (y - np.min(y))/(np.max(y) - np.min(y))


fig = plt.figure(figsize=(7, 7))
ax = [fig.add_subplot(u) for u in (221, 222, 223, 224)]

n = 200
m = 6

X = np.random.rand(n, m)
Y = np.random.rand(n, 2)

net = Net(6, 2)
net.buildWeights()

nX, nY = net.normFunction(X, Y)

for ii, (ix, iy) in enumerate(zip(X, Y)):
    print("Rows left: ", n - ii + 1)
    net(ax, ix, iy)


plt.show()


    
