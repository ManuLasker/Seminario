from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

def f(x):
    return x**2

def df(x):
    return 2*x

x = np.linspace(-10, 10, 101)
y = lambda x: f(x)

fig = plt.figure(figsize=(5,5))
plt.plot(x, y(x))
plt.title('2D gradient descent')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

x_init = -10
X = [x_init]
Y = [y(x_init)]
# Begin Gradient descent with N steps and a learning reate
N = 10
alpha = 0.3

for j in range(N):
    last_X = X[-1]
    this_X = last_X - alpha * df(last_X)
    X.append(this_X)
    Y.append(y(this_X))


theta = []
for j in range(len(X)):
    val_X = X[j]
    val_Y = Y[j]
    theta.append(np.array((val_X, val_Y)))

for j in range(N):
    if j>0:
        plt.annotate('',xy=theta[j], xytext=theta[j-1],
                    arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                    va='center', ha='center')
    plt.scatter(X[j], Y[j], s=40, lw=0)
    plt.pause(0.5)


plt.show()