from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

# The data to fit
m = 20
theta0_true = 2
theta1_true = 0.5
x = np.linspace(-1,1,m)
y = theta0_true + theta1_true * x

# Figure
fig =  plt.figure(figsize=(14,5))

# First plot
ax = fig.add_subplot(1,3,1)

# Scatter plot on the real data
ax.scatter(x, y, marker = 'x', s = 40, color = 'k')

def cost_func(theta0, theta1):
    """The cost function, J(theta0, theta1) describing the goodness of fit."""
    theta0 = np.atleast_3d(np.asarray(theta0))
    theta1 = np.atleast_3d(np.asarray(theta1))
    return np.average((y-hypothesis(x, theta0, theta1))**2, axis=2)/2

def hypothesis(x, theta0, theta1):
    """Our "hypothesis function", a straight line."""
    return theta0 + theta1*x

# First construct a grid of (theta0, theta1) parameter pairs and their
# corresponding cost function values.
theta0_grid = np.linspace(-1,4,101)
theta1_grid = np.linspace(-5,5,101)
J_grid = cost_func(theta0_grid[np.newaxis,:,np.newaxis],
                theta1_grid[:,np.newaxis,np.newaxis])

# Setup for 3d plot or surface plot
X, Y = np.meshgrid(theta0_grid, theta1_grid)

# Contour plot
ax1 = fig.add_subplot(1,3,2)
ax1.set_title('Contour plot')
contours = ax1.contour(X, Y, J_grid, 30)
ax1.clabel(contours)
# The target parameter values indicated on the cost function contour plot
ax1.scatter([theta0_true]*2,[theta1_true]*2,s=[50,10], color=['k','w'])

# Surface Plot
ax2 = fig.add_subplot(1,3,3, projection = '3d')
# ax2.plot_wireframe(X, Y, J_grid, rstride=10, cstride=10,
#                     linewidth=1, antialiased=False, cmap=cm.coolwarm)
norm = plt.Normalize(J_grid.min(), J_grid.max())
colors = cm.viridis(norm(J_grid))
rcount, ccount, _ = colors.shape

surf = ax2.plot_surface(X, Y, J_grid, rcount=10, ccount=10,
                    facecolors=colors, shade=False)
surf.set_facecolor((0,0,0,0))
# plot the real point
ax2.scatter([theta0_true]*2,[theta1_true]*2, cost_func(theta0_true, theta1_true), s=50 ,color=['r'])

# Take N steps with learning rate alpha down the steepest gradient, starting at (0,0)
N = 10
alpha = 0.5

# Initial theta
theta = [np.array((0,0))]
J = [cost_func(*theta[0])[0]]
for j in range(N):
    last_theta = theta[-1]
    this_theta = np.empty((2,))
    this_theta[0] = last_theta[0] - alpha / m * np.sum(
                                    (hypothesis(x, *last_theta) - y))
    this_theta[1] = last_theta[1] - alpha / m * np.sum(
                                    (hypothesis(x, *last_theta) - y) * x)
    theta.append(this_theta)
    J.append(cost_func(*this_theta))

colors = cm.rainbow(np.linspace(0, 1, N+1))
ax.plot(x, hypothesis(x, *theta[0]), color = colors[0], lw = 2, label=r'$\theta_0 = {:.3f}, \theta_1 = {:.3f}$'.format(*theta[j]))

for j in range(0,N):
    if j>0:
        ax1.annotate('',xy=theta[j], xytext=theta[j-1],
                    arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                    va='center', ha='center')
    ax.plot(x, hypothesis(x, *theta[j]), color=colors[j], lw=2,
            label=r'$\theta_0 = {:.3f}, \theta_1 = {:.3f}$'.format(*theta[j]))
    ax1.scatter(theta[j][0], theta[j][1], s=40, lw=0)
    ax2.scatter(theta[j][0], theta[j][1], J[j], s=40, lw = 0)
    plt.pause(0.5)
# Labels, titles and a legend.
ax1.set_xlabel(r'$\theta_0$')
ax1.set_ylabel(r'$\theta_1$')
ax1.set_title('Cost function')
ax2.set_xlabel(r'$\theta_0$')
ax2.set_ylabel(r'$\theta_1$')
ax2.set_zlabel(r'cost')
ax2.set_title('3D plot')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_title('Data and fit')
axbox = ax.get_position()
# Position the legend by hand so that it doesn't cover up any of the lines.
ax.legend(loc=(axbox.x0+0.5*axbox.width, axbox.y0+0.1*axbox.height),
            fontsize='small')
plt.tight_layout()
plt.show()