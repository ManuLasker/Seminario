from math import sqrt, ceil
import numpy as np
import matplotlib.pyplot as plt

def visualize_grid(Xs, ubound=255.0, padding=1):
	"""
	Reshape a 4D tensor of image data to a grid for easy visualization.

	Inputs:
	- Xs: Data of shape (N, H, W, C)
	- ubound: Output grid will have values scaled to the range [0, ubound]
	- padding: The number of blank pixels between elements of the grid
	"""
	(N, H, W, C) = Xs.shape
	grid_size = int(ceil(sqrt(N)))
	grid_height = H * grid_size + padding * (grid_size - 1)
	grid_width = W * grid_size + padding * (grid_size - 1)
	grid = np.zeros((grid_height, grid_width, C))
	next_idx = 0
	y0, y1 = 0, H
	for y in range(grid_size):
		x0, x1 = 0, W
		for x in range(grid_size):
			if next_idx < N:
				img = Xs[next_idx]
				low, high = np.min(img), np.max(img)
				grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
				# grid[y0:y1, x0:x1] = Xs[next_idx]
				next_idx += 1
			x0 += W + padding
			x1 += W + padding
		y0 += H + padding
		y1 += H + padding
	# grid_max = np.max(grid)
	# grid_min = np.min(grid)
	# grid = ubound * (grid - grid_min) / (grid_max - grid_min)
	return grid

def visualize_grid_withoutRGB(Xs, ubound = 255.0, padding = 1):
	N, H, W = Xs.shape
	grid_size = int(ceil(sqrt(N)))
	grid_heigt = H * grid_size + padding * (grid_size - 1)
	grid_width = W * grid_size + padding * (grid_size - 1)
	grid =  np.zeros((grid_heigt, grid_width))
	next_idx = 0
	y0, y1 = 0, H
	for y in range(grid_size):
		x0, x1 = 0, W
		for x in range(grid_size):
			if next_idx < N:
				img = Xs[next_idx]
				low, high = np.min(img), np.max(img)
				grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
				# grid[y0:y1, x0:x1] = Xs[next_idx]
				next_idx += 1
			x0 += W + padding
			x1 += W + padding
		y0 += H + padding
		y1 += H + padding
	# grid_max = np.max(grid)
	# grid_min = np.min(grid)
	# grid = ubound * (grid - grid_min) / (grid_max - grid_min)
	return grid

def vis_grid(Xs):
	""" visualize a grid of images """
	(N, H, W, C) = Xs.shape
	A = int(ceil(sqrt(N)))
	G = np.ones((A*H+A, A*W+A, C), Xs.dtype)
	G *= np.min(Xs)
	n = 0
	for y in range(A):
		for x in range(A):
			if n < N:
				G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = Xs[n,:,:,:]
				n += 1
	# normalize to [0,1]
	maxg = G.max()
	ming = G.min()
	G = (G - ming)/(maxg-ming)
	return G

def vis_nn(rows):
	""" visualize array of arrays of images """
	N = len(rows)
	D = len(rows[0])
	H,W,C = rows[0][0].shape
	Xs = rows[0][0]
	G = np.ones((N*H+N, D*W+D, C), Xs.dtype)
	for y in range(N):
		for x in range(D):
			G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = rows[y][x]
	# normalize to [0,1]
	maxg = G.max()
	ming = G.min()
	G = (G - ming)/(maxg-ming)
	return G

def plotStats(results, verbose = False):
	# Plot the loss function and train / validation accuracies
	# Plot learning curves
	plt.rcParams['figure.figsize'] = (8, 5) # set default size of plots
	if verbose:
		num = 2 + results['ratio_history'].shape[1]
	else:
		num = 2
	plt.subplot(num, 1, 1)
	plt.plot(results['loss_history'])
	plt.title('Loss history')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')

	plt.subplot(num, 1, 2)
	plt.plot(results['train_acc_history'], label='train')
	plt.plot(results['val_acc_history'], label='val')
	plt.title('Classification accuracy history')
	plt.xlabel('Epoch')
	plt.ylabel('Clasification accuracy')
	plt.legend()


	if verbose == True:
		for i in range(results['ratio_history'].shape[1]):
			plt.subplot(num, 1, i+3)
			plt.plot(results['ratio_history'][:,i])
			plt.xlabel('Epoch')
			plt.ylabel('Ratio of gradient change')
	plt.tight_layout()
	plt.show()

def plotData(X, y, y_pred):
	plt.subplot(2, 1, 1)
	plt.title('real')
	cross, = plt.plot([],[],'rx')
	good, = plt.plot([],[],'go')
	for i in range(len(y)):
		if y[i] == 0:
			plt.plot(X[i,0], X[i,1], 'rx')
		else:
			plt.plot(X[i,0],X[i,1], 'go')
	plt.legend(handles = [cross, good], labels = ['False', 'True'],bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
	plt.tight_layout()

	plt.subplot(2, 1, 2)
	plt.title('predicción')
	for i in range(len(y_pred)):
		if y_pred[i] == 0:
			plt.plot(X[i,0], X[i,1], 'rx')
		else:
			plt.plot(X[i,0],X[i,1], 'go')

	plt.legend(handles = [cross, good], labels = ['False', 'True'],bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
	plt.tight_layout()
	plt.show()



