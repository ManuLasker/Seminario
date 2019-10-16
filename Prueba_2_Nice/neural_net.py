import numpy as np
from tqdm import tqdm_notebook as tqdm
import time
#from tqdm import tqdm
class neural_net():

	def __init__(self, net_params, activation_function = 'ReLu', \
		cost_function = 'SoftMax', std = 1e-4):
		self.net_params = net_params
		self.std = std
		self.function_params = {}
		self.activation_function = activation_function
		self.cost_function = cost_function
		if self.validate_net() == 0:
			return
		"""
		input_size will contain the size of the input layer
		output_size will contain the size of the ouput layer
		hidden_params will contain the information of each hidden layer
		for example:
			hidden_params = {'hl1': hl1_size, 'hl2': hl2_size, ..., 'hln': hln}
			len(hidden_params) = will be the number of hidden layers

			the net structure is the follow:
				il -> hl1 -> ... -> hl3 -> ol
		"""
		# get the last key of the dictionary
		self.net_params, self.function_params = self.InitializeNetFunctionParams(net_params, std)

	def validate_net(self):
		if self.activation_function == 'ReLu' and self.cost_function == 'SoftMax':
			if self.net_params['ol'] == 1:
				print('ol_size must be greater than 2')
				return 0

	def InitializeNetFunctionParams(self, net_params, std = 1e-4):
		num_weights = len(net_params)
		last_hidden_output = False
		i = 1
		while i < num_weights:
			if  i == 1:
				if 'hl1' not in list(self.net_params.keys()):
					# Does not have hidden layers just output layer
					# i_layer --- o_layer
					key = 'ol'
				else:
					key = 'hl1'
				C_out = self.net_params[f'{key}']
				C_in = self.net_params[f'il']
				self.function_params[f'W{i}'] = self.randInitializeWeigths(C_in, C_out, std)
				self.function_params[f'b{i}'] = self.randInitializeBiases(C_out, std)
			elif i == (num_weights - 1):
				C_in = self.net_params[f'hl{i-1}']
				C_out = self.net_params[f'ol']
				self.function_params[f'W{i}'] = self.randInitializeWeigths(C_in, C_out, std)
				self.function_params[f'b{i}'] = self.randInitializeBiases(C_out, std)
			elif i > 0 and i < num_weights:
				C_in = self.net_params[f'hl{i-1}']
				C_out = self.net_params[f'hl{i}']
				self.function_params[f'W{i}'] = self.randInitializeWeigths(C_in, C_out, std)
				self.function_params[f'b{i}'] = self.randInitializeBiases(C_out, std)
			i += 1
		return self.net_params, self.function_params

	def randInitializeWeigths(self, C_in, C_out, std = 1e-4):
		"""
		Initialize the weights in a random way
		The weights will be a matrix of shape [C_in, C_out]
		C_in = incomming connection
		C_out = outcomming connection
		For example:
			layer1 -> layer2
		C_in = layer1_size
		C_out = layer2_size
		"""
		# epsilon = 0.12
		# weights = epsilon*(np.random.randn(C_in, C_out)*2 - 1)
		# weights = np.ones((C_in, C_out))
		# weights =std*np.random.randn(C_in, C_out)
		if self.activation_function == 'ReLu':
			weights = std * np.random.randn(C_in, C_out)/np.sqrt(C_in/2)
		elif self.activation_function == 'Sigmoid':
			weights = std * np.random.randn(C_in, C_out)/np.sqrt(C_in)
		return weights

	def randInitializeBiases(self, C_out, std = 1e-4):
		"""
		Initialize the biases in a random way
		The bias will be an array of size [C_out]
		C_out = outcomming connection
		For example:
			layer1 -> layer2
		C_out = layer2_size
		"""
		# epsilon = 0.12
		# bias = epsilon*(np.random.randn(1, C_out)*2 - 1)
		# bias = np.ones((1, C_out))
		# bias = np.zeros(C_out)
		if self.activation_function == 'ReLu':
			bias =std * np.random.randn(C_out)/np.sqrt(1/2)
		elif self.activation_function == 'Sigmoid':
			bias =std * np.random.randn(C_out)
		return bias

	def ReLu(self, z):
		"""
		Return ReLu function evaluated at z, f(z) = max(0, z)

		"""
		r = np.maximum(0, z)
		return r

	def sigmoid(self, z):
		"""
		Return the sigmoid function evaluated at z
		"""
		g = 1/(1 + np.exp(-z))
		return g

	def feed_forward(self, X):
		"""
		Calculate all the feed forward score in each step
		"""
		forward_results = {}

		if self.activation_function == 'Sigmoid':
			"""
			sigmoid forward pass:
				input -> sigmoid(x*w + b) -> output
			"""
			forward_results['flc1'] = self.sigmoid(X.dot(self.function_params['W1']) - self.function_params['b1'])
			for i in range(1, len(self.net_params) - 1):
				forward_results[f'flc{i+1}'] = self.sigmoid(forward_results[f'flc{i}'].dot(self.function_params[f'W{i+1}']) \
					- self.function_params[f'b{i+1}'])
		elif self.activation_function == 'ReLu':
			"""
			Use the Relu
			"""
			forward_results['flc1'] = self.ReLu(X.dot(self.function_params['W1']) + self.function_params['b1'])
			for i in range(1, len(self.net_params) - 1):
				if i == (len(self.net_params) - 2):
					forward_results[f'flc{i+1}'] = forward_results[f'flc{i}'].dot(self.function_params[f'W{i+1}']) \
						+ self.function_params[f'b{i+1}']
				else:
					forward_results[f'flc{i+1}'] = self.ReLu(forward_results[f'flc{i}'].dot(self.function_params[f'W{i+1}']) \
						+ self.function_params[f'b{i+1}'])
		self.forward_results = forward_results
		last_key = list(self.forward_results.keys())[-1]
		return self.forward_results[last_key]

	def loss_function(self, X, y, reg = 0.0):
		"""
		Calculate the loss or performance of the net
		"""
		loss = 0
		if self.cost_function == 'Entropy_Loss':
			if 'ol' in list(self.net_params.keys()):
				o_size = self.net_params['ol']
				if o_size != 1:
					I_matrix = np.eye(o_size)
					y = I_matrix[y.reshape(len(y))]

			score = self.feed_forward(X)
			# print(y)
			loss =  np.sum(-y * np.log(score) - (1 - y) * np.log(1 - score))
			add_reg = 0
			for key in self.function_params:
				if key[0] != 'b':
					add_reg += np.sum(self.function_params[key]**2)
			loss += 1/2 * reg * add_reg
			loss *= 1/X.shape[0]
		elif self.cost_function == 'SoftMax':
			N = X.shape[0]
			scores = self.feed_forward(X).copy()
			scores -= np.max(scores, axis=1, keepdims=True) # avoid numeric instability
			scores_exp = np.exp(scores) # exp all the element of the score matrix
			softmax_matrix = scores_exp / np.sum(scores_exp, axis=1, keepdims=True) # softmax

			if 'ol' in list(self.net_params.keys()):
				o_size = self.net_params['ol']
				if o_size != 1:
					I_matrix = np.eye(o_size)
					y = I_matrix[y.reshape(len(y))]

			softmax_temp = np.sum(softmax_matrix*y, axis = 1, keepdims = True) # Choos the right classes
			# loss = np.sum(-np.log(softmax_matrix[np.arange(N), y.reshape(len(y))]))
			loss = np.sum(-np.log(softmax_temp))
			loss /= N
			add_reg = 0
			for key in self.function_params:
				if key[0] != 'b':
					add_reg += np.sum(self.function_params[key]**2)
			loss += reg * add_reg# regularization
		return loss

	def back_prop(self, X, y, reg = 0.0):
		"""
		Return the grads of the performance function for each weights
		"""
		grads = {}
		local_grads = {}
		activation_function_grads = {}
		y_temp = y
		N, D = X.shape
		if self.cost_function == 'Entropy_Loss' and self.activation_function == 'Sigmoid':
			if 'ol' in list(self.net_params.keys()):
				o_size = self.net_params['ol']
				if o_size != 1:
					I_matrix = np.eye(o_size)
					y_temp = I_matrix[y.reshape(len(y))]
			z = self.feed_forward(X)
			n_grads = len(self.forward_results)
			loss_grad = (z-y_temp)/(z*(1-z))
			i = n_grads
			mult_const = 1/X.shape[0]

			while i > 0:
				if i == n_grads:
					activation_function_grads[f'dS{i}'] = 	self.forward_results[f'flc{i}'] \
						* (1 - self.forward_results[f'flc{i}'])
					local_grads[f'dL{i}'] = loss_grad * activation_function_grads[f'dS{i}']
				else:
					activation_function_grads[f'dS{i}'] = self.forward_results[f'flc{i}'] \
						* (1 - self.forward_results[f'flc{i}'])
					local_grads[f'dL{i}'] = local_grads[f'dL{i + 1}'] @ self.function_params[f'W{i + 1}'].T \
						* activation_function_grads[f'dS{i}']

				if i==1:
					grads[f'W{i}'] =mult_const* (X.T.dot(local_grads[f'dL{i}']) + reg*self.function_params[f'W{i}'])
					grads[f'b{i}'] = mult_const* (-np.sum(local_grads[f'dL{i}'], axis = 0))
				else:
					grads[f'W{i}'] =mult_const* (self.forward_results[f'flc{i - 1}'].T.dot(local_grads[f'dL{i}']) \
						+ reg*self.function_params[f'W{i}'])
					grads[f'b{i}'] =mult_const* (-np.sum(local_grads[f'dL{i}'], axis = 0))
				i -= 1
		elif self.cost_function == 'SoftMax' and self.activation_function == 'ReLu':
			if 'ol' in list(self.net_params.keys()):
				o_size = self.net_params['ol']
				if o_size != 1:
					I_matrix = np.eye(o_size)
					y_temp = I_matrix[y.reshape(len(y))]
			scores = self.feed_forward(X).copy()
			n_grads = len(self.forward_results)
			scores -= np.max(scores, axis=1, keepdims=True) # avoid numeric instability
			scores_exp = np.exp(scores) # exp all the element of the score matrix
			softmax_matrix = scores_exp / np.sum(scores_exp, axis=1, keepdims=True) # softmax

			softmax_matrix_temp = softmax_matrix * y_temp # the softmax of the true
			softmax_matrix_temp2 = softmax_matrix * (y_temp == 0).astype(int) # the softmax of the non-true classes
			loss_softmax_grad = (-1 + softmax_matrix_temp)*y_temp + (softmax_matrix_temp2) # loss grad
			loss_softmax_grad /= N
			i = n_grads
			while i > 0:
				if i == n_grads:
					activation_function_grads[f'dS{i}'] = loss_softmax_grad
					local_grads[f'dL{i}'] = activation_function_grads[f'dS{i}']
				else:
					activation_function_grads[f'dS{i}'] = self.forward_results[f'flc{i}'] != 0
					local_grads[f'dL{i}'] = local_grads[f'dL{i + 1}'] @ self.function_params[f'W{i + 1}'].T \
						* activation_function_grads[f'dS{i}']
				if i==1:
					grads[f'W{i}'] =(X.T.dot(local_grads[f'dL{i}']) + 2*reg*self.function_params[f'W{i}'])
					grads[f'b{i}'] =(np.sum(local_grads[f'dL{i}'], axis = 0))
				else:
					grads[f'W{i}'] =(self.forward_results[f'flc{i - 1}'].T.dot(local_grads[f'dL{i}']) + \
						2*reg*self.function_params[f'W{i}'])
					grads[f'b{i}'] =(np.sum(local_grads[f'dL{i}'], axis = 0))
				i -= 1
		self.grads = grads
		return grads

	def train_gradient_descent(self, X, y, X_val, y_val, \
		learning_rate = 1e-3, max_iter = 400, batch_size = 200, \
		verbose = False, reg = 0.0, learning_rate_decay = 0.999, epoch = 20):
		"""
		Use Gradient descent to train and optimize the net
		"""
		num_train = X.shape[0]
		# iterations_per_epoch = max(num_train / batch_size, 1)
		loss_history = []
		train_acc_history = []
		val_acc_history = []
		ratio = []
		iteration = 0
		loss_ant = 0
		loss_post = 0
		num_update_weigths = 0
		with tqdm(total=max_iter) as pbar:
			while iteration < (max_iter):
				grads = self.back_prop(X, y, reg = reg)
				loss = self.loss_function(X, y, reg = reg)
				loss_ant = loss_post
				loss_post = loss

				if np.isnan(loss) or np.isposinf(loss):
					print('#### LOSS EXPLOTE #### \n')
					break
				loss_history.append(loss)
				for key in grads:
					self.function_params[key] -= learning_rate*grads[key]

				if verbose:
					#print(f'iteration {iteration} / {max_iter}, loss {loss.round(decimals = 4)} {iteration/max_iter * 100}', end = '\r')
					pbar.set_description(f'iteration {iteration} / {max_iter}, loss {loss.round(decimals = 4)}')
					pbar.update(1)

				if iteration % epoch == 0:
					# Check accuracy
					train_acc = (self.predict(X) == y).mean()
					val_acc = (self.predict(X_val) == y_val).mean()
					train_acc_history.append(train_acc)
					val_acc_history.append(val_acc)
					ratio_temp = list(self.ratio_weigths(self.function_params, learning_rate, grads))
					ratio.append(ratio_temp)
					# Decay learning rate
					if round(loss_ant/loss_post, 4) == 1:
						self.net_params, self.function_params = self.InitializeNetFunctionParams(self.net_params, self.std)
						iteration = 0
						num_update_weigths += 1

					if num_update_weigths == 500:
						print('\n ###### Model get Stuck ######')
						break
					learning_rate *= learning_rate_decay
				iteration += 1
		return {
		'loss_history': loss_history,
		'train_acc_history': train_acc_history,
		'val_acc_history': val_acc_history,
		'ratio_history': np.array(ratio),
		}

	def predict(self, X, with_score = False):
		N, D = X.shape
		scores = self.feed_forward(X).copy()
		if 'ol' in list(self.net_params.keys()):
			if self.activation_function == 'Sigmoid':
				o_size = self.net_params['ol']
				if o_size == 1:
					predict = scores.round()
				else:
					predict = np.argmax(scores, axis = 1)
			elif self.activation_function == 'ReLu':
				o_size = self.net_params['ol']
				# scores -= np.max(scores, axis=1, keepdims=True) # avoid numeric instability
				scores_exp = np.exp(scores)
				softmax_matrix = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
				scores = softmax_matrix 
				if o_size == 1:
					predict = scores.round()
				else:
					predict = np.argmax(scores, axis = 1)
		if with_score:
			return predict, scores
		else:
			return predict

	def zero_mean_data(self, X):
		mean_X = np.mean(X, axis = 1, keepdims = True)
		X = X - mean_X
		return X

	def train_sgd_momentum(self, X, y, X_val, y_val,learning_rate=1e-3,\
		learning_rate_decay=0.95,reg=5e-6,\
		max_iter=100, batch_size=200, verbose=False, rho = 0.99, stochastic = True, epoch = 20):

		num_train = X.shape[0]
		# iterations_per_epoch = max(num_train / batch_size, 1)

		# Use SGD to optimize the parameters in self.model
		loss_history = []
		train_acc_history = []
		val_acc_history = []
		ratio = []
		vx = {}
		it = 0
		loss_ant = 0
		loss_post = 0
		num_update_weigths = 0
		# Initialize velocity to 0
		for key in list(self.function_params.keys()):
			vx[key] = 0

		with tqdm(total=max_iter) as pbar:
			while it < max_iter:
				X_batch = None
				y_batch = None
				if stochastic:
					batch_indices = np.random.choice(num_train, batch_size)
					X_batch = X[batch_indices]
					y_batch = y[batch_indices]
				else:
					X_batch = X
					y_batch = y

				# Compute loss and gradients using the current minibatch
				loss =  self.loss_function(X_batch, y=y_batch, reg = reg)
				grads = self.back_prop(X_batch, y=y_batch, reg=reg)
				loss_ant = loss_post
				loss_post = loss

				loss_history.append(loss)
				if np.isnan(loss) or np.isposinf(loss):
					print('#### LOSS EXPLOTE #### \n')
					break

				for key in self.function_params:
					vx[key] = rho * vx[key] + grads[key]
					self.function_params[key] -= learning_rate * vx[key]

				if verbose:
					pbar.set_description(f'iteration {it} / {max_iter}, loss {loss.round(decimals = 4)}')
					pbar.update(1)

				# Every epoch, check train and val accuracy and decay learning rate.
				if it % epoch == 0:
					# Check accuracy
					#print(f'{it}, {loss}', end = '\r')
					train_acc = (self.predict(X_batch) == y_batch).mean()
					val_acc = (self.predict(X_val) == y_val).mean()
					train_acc_history.append(train_acc)
					val_acc_history.append(val_acc)
					ratio_temp = list(self.ratio_weigths(self.function_params, learning_rate, grads))
					ratio.append(ratio_temp)
					# Decay learning rate
					learning_rate *= learning_rate_decay
					if stochastic == False:
						if round(loss_ant/loss_post, 4) == 1:
							self.net_params, self.function_params = self.InitializeNetFunctionParams(self.net_params, self.std)
							it = 0
							num_update_weigths += 1

						if num_update_weigths == 500:
							print('\n ###### Model get Stuck ######')
							break
				it += 1
		return {
		'loss_history': loss_history,
		'train_acc_history': train_acc_history,
		'val_acc_history': val_acc_history,
		'ratio_history': np.array(ratio),
		}

	def train_stochastic_gradient_descent(self, X, y, X_val, y_val,learning_rate=1e-3,\
		learning_rate_decay=0.95,reg=5e-6,\
		max_iter=100, batch_size=200, verbose=False, epoch = 20):

		num_train = X.shape[0]
		# iterations_per_epoch = max(num_train / batch_size, 1)

		# Use SGD to optimize the parameters in self.model
		loss_history = []
		train_acc_history = []
		val_acc_history = []
		ratio = []
		with tqdm(total=max_iter) as pbar:
			for it in range(max_iter):
				X_batch = None
				y_batch = None

				batch_indices = np.random.choice(num_train, batch_size)
				X_batch = X[batch_indices]
				y_batch = y[batch_indices]

				# Compute loss and gradients using the current minibatch
				loss =  self.loss_function(X_batch, y=y_batch, reg = reg)
				grads = self.back_prop(X_batch, y=y_batch, reg=reg)
				# Append ratio of weights
				loss_history.append(loss)
				if np.isnan(loss) or np.isposinf(loss):
					print('#### LOSS EXPLOTE #### \n')
					break
				for key in self.function_params:
					self.function_params[key] -= learning_rate * grads[key]

				if verbose:
					pbar.set_description(f'iteration {it} / {max_iter}, loss {loss.round(decimals = 4)}')
					pbar.update(1)
				# Every epoch, check train and val accuracy and decay learning rate.
				if it % epoch == 0:
					# Check accuracy
					#print(f'{it}, {loss}', end = '\r')
					train_acc = (self.predict(X_batch) == y_batch).mean()
					val_acc = (self.predict(X_val) == y_val).mean()
					train_acc_history.append(train_acc)
					val_acc_history.append(val_acc)
					ratio_temp = list(self.ratio_weigths(self.function_params, learning_rate, grads))
					ratio.append(ratio_temp)

					# Decay learning rate
					learning_rate *= learning_rate_decay

		return {
		'loss_history': loss_history,
		'train_acc_history': train_acc_history,
		'val_acc_history': val_acc_history,
		'ratio_history': np.array(ratio),
		}

	def ratio_weigths(self, param, learning_rate, grad_param):
		param_scale = []
		update_scale = []
		for key in list(param.keys()):
			op1 = np.linalg.norm(param[key].ravel())
			param_scale.append(op1)
			update = -learning_rate*grad_param[key]
			update_scale.append(np.linalg.norm(update.ravel()))
		param_scale = np.array(param_scale)
		update_scale = np.array(update_scale)
		return (param_scale/update_scale)
