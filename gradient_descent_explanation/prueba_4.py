from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

X = np.array([[1,1],
              [0,1],
              [1,0],
              [0,0]])

y = np.array([0,1,1,0])
fig = plt.figure(figsize=(10,5))
ax0 = fig.add_subplot(1,2,1)
ax0.set_title('real')
ax0.set_xlabel('x')
ax0.set_ylabel('y')
ax0.set_xlim(-0.5,1.5)
ax0.set_ylim(-0.5,1.5)
cross, = ax0.plot([],[],'rx')
good, = ax0.plot([],[],'go')
for i in range(len(y)):
    if y[i] == 0:
        ax0.plot(X[i,0], X[i,1], 'rx')
    else:
        ax0.plot(X[i,0],X[i,1], 'go')
ax0.legend(handles = [cross, good], 
           labels = ['False', 'True'],bbox_to_anchor=(1.05, 1),
           loc='upper left', borderaxespad=0)
plt.tight_layout()

ax = fig.add_subplot(1,2,2, projection = '3d')
ax.set_title('3D gradient descent plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(-1.5,1.5)

x1 = np.linspace(-0.5,1.5,101)
y1 = np.linspace(-0.5,1.5,101)
w1 = np.array([[-0.5, 1.5],
               [1, -1],
               [1, -1]])

w2 = np.array([-1.5, 1, 1])

X1,Y1 = np.meshgrid(x1, y1)

Z = ((w1[1,0]*X1+w1[2,0]*Y1+w1[0,0]) > 0).astype(int) *w2[1] + ((w1[1,1]*X1+w1[2,1]*Y1+w1[0,1]) > 0).astype(int)*w2[2] + w2[0]

norm = plt.Normalize(Z.min(), Z.max())
colors = cm.viridis(norm(Z))
rcount, ccount, _ = colors.shape
surf = ax.plot_surface(X1, Y1, Z,rcount=20, ccount=20,
                    facecolors=colors, shade=False )
surf.set_facecolor((0,0,0,0))
cross, = ax.plot([],[],'rx')
good, = ax.plot([],[],'go')
for i in range(len(y)):
    if y[i] == 0:
        ax.scatter(X[i,0], X[i,1], 0, marker ='x', c='red')
    else:
        ax.scatter(X[i,0],X[i,1], 0, marker='o', c='green')
ax.legend(handles = [cross, good], 
           labels = ['False', 'True'],bbox_to_anchor=(1.05, 1),
           loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.show()