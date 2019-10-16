import numpy as np

def gradient_difference(net, X, y, grads, reg = 0.0):
	grad = {}
	for param_name in net.function_params:
		fun = lambda W: net.loss_function(X, y, reg)
		grad[param_name] = checkGradient(fun, net.function_params[param_name])
	for key in grad:
		print(f'difference {key}', np.linalg.norm(grad[key] - grads[key]))

def checkGradient(f, x, verbose=False, h=0.00001):
	fx = f(x) # evaluate function value at original point
	grad = np.zeros_like(x)
	# iterate over all indexes in x
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:

		# evaluate function at x+h
		ix = it.multi_index
		oldval = x[ix]
		x[ix] = oldval + h # increment by h
		fxph = f(x) # evalute f(x + h)
		x[ix] = oldval - h
		fxmh = f(x) # evaluate f(x - h)
		x[ix] = oldval # restore

		# compute the partial derivative with centered formula
		grad[ix] = (fxph - fxmh) / (2 * h) # the slope
		if verbose:
			print(ix, grad[ix])
		it.iternext() # step to next dimension

	return grad