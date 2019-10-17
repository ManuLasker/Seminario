from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

def f(x,y):
    return x**2 + y**2

def dfdy(x,y):
    return 2*y

def dfdx(x,y):
    return 2*x

fig =  plt.figure(figsize=(10,5))
ax0 =  fig.add_subplot(1,2,1)
ax0.set_title('contour plot')
ax0.set_xlabel('x')
ax0.set_ylabel('y')

ax = fig.add_subplot(1,2,2, projection = '3d')
ax.set_title('3D gradient descent plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

x = np.linspace(-10,10,101)
y = np.linspace(-10,10,101)

X,Y = np.meshgrid(x, y)
Z = f(X,Y)
norm = plt.Normalize(Z.min(), Z.max())
colors = cm.viridis(norm(Z))
rcount, ccount, _ = colors.shape
surf = ax.plot_surface(X, Y, Z,rcount=20, ccount=20,
                    facecolors=colors, shade=False )
surf.set_facecolor((0,0,0,0))
contours = ax0.contour(X, Y, Z, 30)
ax0.clabel(contours)

# Take N steps with learning rate alpha down the steepest gradient, starting at (0,0)
N = 10
alpha = 0.2
p_init = (-5,10)
P = [np.array(p_init)]
F = [f(*P[0])]

for i in range(N):
    last_P = P[-1]
    new_P = np.zeros(2)
    new_P[0] = last_P[0] - alpha*dfdx(last_P[0], last_P[1])
    new_P[1] = last_P[1] - alpha*dfdy(last_P[0], last_P[1])
    P.append(new_P)
    F.append(f(new_P[0], new_P[1]))

colors = cm.rainbow(np.linspace(0, 1, N+1))


for j in range(0,N):
    if j>0:
        ax0.annotate('',xy=P[j], xytext=P[j-1],
                    arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                    va='center', ha='center')
    ax0.scatter(P[j][0], P[j][1], s=40, lw=0)
    ax.scatter(P[j][0], P[j][1], F[j], s=40, lw = 0)
    plt.pause(0.8)

plt.show()