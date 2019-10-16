import numpy as np
from tqdm import tqdm
from neural_net import neural_net

def tuning_hyper_parameter(net_params, X, y, X_val, y_val,\
	h_min = 1, h_max = 2, reg_min = -7, reg_max = 2, \
	lr_min = -6, lr_max = 1,\
	max_count = 20, epoch = 20, max_iter = 400,\
	method = 'gd', batch_size = 1, learning_decay = 1, hidden_layers = 1,\
	activation_function = 'ReLu', cost_function = 'SoftMax'):

	best_net = None
	best_val = -1
	best_results = -1
	for proof in range(max_count):
		net_params, lr, reg = generate_random_hyperparams(lr_min, lr_max, reg_min,\
			reg_max, h_min, h_max, net_params, hidden_layers)
		net =  neural_net(net_params, activation_function, cost_function, std = 1)

		if method == 'gd':
			print('\n Tuning with gradient descent')
			results = net.train_gradient_descent(X, y, X_val, y_val, learning_rate = lr,\
				max_iter = max_iter, batch_size = batch_size, verbose = True, \
				reg = reg, learning_rate_decay = learning_decay, epoch = epoch)
		elif method == 'sgd':
			print('\n Tuning with stochastic gradient descent')
			results = net.train_stochastic_gradient_descent(X, y, X_val, y_val, learning_rate = lr,\
				max_iter = max_iter, batch_size = batch_size, verbose = True, \
				reg = reg, learning_rate_decay = learning_decay, epoch = epoch)
		elif method == 'sgd+momentum':
			print('\n Tuning with stochastic gradient descent + momentum')
			results = net.train_sgd_momentum( X, y, X_val, y_val,learning_rate=lr,\
				learning_rate_decay=learning_decay,reg=reg,\
				max_iter=max_iter, batch_size=batch_size, verbose=True,\
				rho = 0.9, stochastic = True, epoch = epoch)
		elif method == 'gd+momentum':
			print('Tuning with gradient descent + momentum')
			results = net.train_sgd_momentum( X, y, X_val, y_val,learning_rate=lr,\
				learning_rate_decay=learning_decay,reg=reg,\
				max_iter=max_iter, batch_size=batch_size, verbose=True,\
				rho = 0.9, stochastic = False, epoch = epoch)

		y_pred_train = net.predict(X)
		train_accuracy = (y_pred_train == y).mean()

		y_pred_val = net.predict(X_val)
		val_accuracy = (net.predict(X_val) == y_val).mean()

		loss = net.loss_function(X, y, reg= reg)
		if val_accuracy > best_val:
			best_val = val_accuracy
			best_lr = lr
			best_reg = reg
			best_net = net
			best_net_params = net_params
			best_results = results
		print(f'{net_params}')
		print(f'lr {round(lr, 5)} reg {round(reg, 5)} loss {round(loss, 5)} train accuracy: {train_accuracy} val accuracy: {val_accuracy}')
	print('\n best validation accuracy achieved: %f' % best_val)
	return best_net, net_params, best_lr, best_reg, best_results
def generate_random_hyperparams(lr_min, lr_max, reg_min, reg_max, h_min, h_max, net_params,\
	hidden_layers):
	for n_hidden in range(1, hidden_layers + 1):
		hidden_dim = np.random.randint(h_min, h_max)
		net_params[f'hl{n_hidden}'] = hidden_dim

	lr = 10**np.random.uniform(lr_min, lr_max)
	reg = 10**np.random.uniform(reg_min, reg_max)

	return net_params, lr, reg


