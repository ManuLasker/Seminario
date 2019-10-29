
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

y = np.array([1,0,0,1])

params = {'W1': np.array([[ 6.03878347, -6.70829504],
        [ 6.03877163, -6.70825261]]),
 'b1': np.array([ 9.33092666, -2.72624356]),
 'W2': np.array([[11.97407601],
        [12.07598502]]),
 'b2': np.array([5.8718201])}

def sigmoid( z):
	"""
	Return the sigmoid function evaluated at z
	"""
	g = 1/(1 + np.exp(-z))
	return g

w = params
x1 = np.linspace(-1.5,1.5,101)
y1 = np.linspace(-1.5,1.5,101)

w1 = w['W1']
b1 = w['b1']
w2 = w['W2']
b2 = w['b2']


X1,Y1 = np.meshgrid(x1, y1)

print(sigmoid(sigmoid(X@w1 - b1)@w2 - b2))
print(b1[0])

Z = sigmoid(sigmoid(X1*w1[0,0]+ Y1*w1[1,0] - b1[0])*w2[0,0] + sigmoid(X1*w1[0,1]+ Y1*w1[1,1] - b1[1])*w2[1,0] - b2[0])



fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1,1,1, projection = '3d')
ax.set_title('3D HyperPlane decision boundary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


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

