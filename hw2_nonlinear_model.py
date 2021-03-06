import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.legend_handler import HandlerLine2D
from numpy.linalg import inv

#finds weights line of best fit with sum of polynomials to the M: y = w * x^0 + w * x^1 ....  + w * x^m
def closed_line_fit_polynomial(input_data,target_data,M,lmbda,num_samp):
	dimensions = input_data.shape
	rows = dimensions[0]
	phi_train = np.ones((rows,M+1))
	# todo make it accept m columbs for different dimensions
	for i in range(rows):
		l = input_data[i]
		phi = np.zeros(M+1)
		for j in range(M+1):
			phi[j] = l**j
		phi_train[i] = phi[:]

	#w_train = np.matmul(np.matmul(inv(np.matmul(phi_train.transpose(),phi_train)) ,phi_train.transpose()), input_data.transpose())#/*+ lmbda * np.identity(M+1)*/
	w_train = np.matmul(np.linalg.pinv(phi_train),target_data)
	return w_train
def creat_line_from_weights(number_of_samples,weights):
	dimensions = weights.shape
	x=np.linspace(0,1,number_of_samples)
	best_fit_line= np.zeros(number_of_samples)
	for i in range(dimensions[0]):
		best_fit_line+= weights[i]*(x**(i))
	return best_fit_line
def error_rms(weights,samples,targets):
	num_samp    = samples.shape
	num_weights = weights.shape
	error = 0
	
	line_guesses = np.zeros(num_samp[0])
	for i in range(num_samp[0]):
		line_guess = 0
		for j in range(num_weights[0]):
			line_guess += weights[j]*samples[i]**j
		line_guesses[i] = line_guess
		error += (line_guess - targets[i])**2
	error = (error/num_samp)**.5
	print(weights)
	plt.figure()
	plt.plot(samples,targets, 'go',samples,line_guesses,'r.')
	plt.show()
	val = input("press enter to continue")
	return error
def E_rms(target_vector, guess_vector):
	dimensions = target_vector.shape
	error = 0
	for i in range(dimensions[0]):
		error += (guess_vector[i] - target_vector[i] )**2
	error = (error**.5)/dimensions[0]
	return error


# dimensions
D = 1

#------------------- generate training set --------------------------
# number of training samples
ntrain = 10

# set seed so that we have same distribution every time
np.random.seed(0)

# sample nt samples from a uniform distribution
train_samples = np.random.uniform(0,1,ntrain)

# design matrix
X_train = np.ones((ntrain, D+1))

for i in range(ntrain):
	X_train[i,0] = train_samples[i]

# noise from gausian distribution with mean = 0 standard deviation = .3
mu, sigma = 0, .3
e = np.random.normal(mu, sigma, ntrain)

# target vector with error
t_train = np.sin(2*math.pi*X_train[:,0]) + e



#------------------generate test set -------------------------------
# number of testing samples
ntest = 100

# reseed for funzies
np.random.seed(2)

# sample ntest samples from niform distribution
test_samples = np.random.uniform(0,1,ntest)

#design matrix for testing
X_test = np.ones((ntest, D+1))

for i in range(ntest):
	X_test[i,0] = test_samples[i]

# noise from gausian distribution with mean = 0 standard deviation = .3
e_test = np.random.normal(mu, sigma, ntest)

t_test = np.sin(2*math.pi*X_test[:,0]) + e_test

#-------- make into nonlinear model ----------
# weight decay factor, reduces higher order terms for line fitting
lambbda  = 0 

# degree of polynomial
M_max = 10
# make line smoother by over making sample size larger
sm = 10
training_error = np.zeros(M_max)
testing_error  = np.zeros(M_max)
train_weights = np.zeros((M_max,M_max))
test_weights  = np.zeros((M_max,M_max))
output_line_train = np.zeros((ntrain*sm,M_max))
output_line_test  = np.zeros((ntest*sm,M_max))

for i in range(M_max):
	train_weights[i] = np.pad(closed_line_fit_polynomial(train_samples,t_train,i,lambbda,ntrain),(0,M_max-i-1),'constant')
	test_weights[i]  = np.pad(closed_line_fit_polynomial(test_samples,t_test,i,lambbda,ntest),(0,M_max-i-1),'constant')
	output_line_test[:,i]  = creat_line_from_weights(ntest*sm,train_weights[i])
	output_line_train[:,i] = creat_line_from_weights(ntrain*sm,train_weights[i])
	training_error[i] = error_rms(train_weights[i],train_samples,t_train)
	testing_error[i]  = error_rms(train_weights[i],test_samples,t_test)
	#training_error[i]  = E_rms(t_train , output_line_train[:,i])
	#testing_error[i]   = E_rms(t_test , output_line_test[:,i])
M = 5
phi_train = np.ones((ntrain,M))
for i in range(ntrain):
	l = X_train[i,0]
	phi = np.zeros(M)
	for j in range(M):
		phi[j] = l**j
	phi_train[i] = phi[:]


w_train = np.matmul(inv(np.matmul(phi_train.transpose(),phi_train)+ lambbda * np.identity(M)) ,np.matmul(phi_train.transpose(), train_samples))

x=np.linspace(0,1,ntrain*sm)
y=w_train[0]*train_samples**0
for i in range(M-1):
	y+= w_train[i+1]*train_samples**(i+1)
# for i in range(M-1):
# 	y += w_train[M] * x**M

# perfect target
p_target = np.sin(2*math.pi*x)

plt.figure()
line1, = plt.plot(training_error, marker='o', label='training_error')
line2, = plt.plot(testing_error, marker='o', label='testing_error')

plt.legend()
plt.show()

for i in range(M_max):
	plt.figure()
	#plt.subplot(121)
	plt.plot(train_samples,t_train, 'go',x,output_line_train[:,i], 'm',x,p_target,'y')
	plt.title('Closed Form M = ' + str(i))
	plt.ylabel('sin with noise')
	plt.xlabel('random x values from uniform distribution')
	plt.show()








