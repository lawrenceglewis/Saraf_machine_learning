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

	w_train = np.matmul(inv(np.matmul(phi_train.transpose(),phi_train)+ lmbda * np.identity(M+1)) ,np.matmul(phi_train.transpose(), input_data))
	#todo make linespace automatic
	# x=np.linspace(0,1,num_samp)
	# # print(X_train[0])
	# # print(train_samples)
	# best_fit_line=w_train[0]*x**0
	# for i in range(M):
	# 	best_fit_line+= w_train[i+1]*x**(i+1)
	return best_fit_line

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
nt = 10

# set seed so that we have same distribution every time
np.random.seed(0)

# sample nt samples from a uniform distribution
train_samples = np.random.uniform(0,1,nt)




# design matrix
X_train = np.ones((nt, D+1))

for i in range(nt):
	X_train[i,0] = train_samples[i]

# noise from gausian distribution with mean = 0 standard deviation = .3
mu, sigma = 0, .3
e = np.random.normal(mu, sigma, nt)
#X = np.squeeze(X)
# print(X_train[:,0])
# target vector with error
t_train = np.sin(2*math.pi*X_train[:,0]) + e

# perfect target
p_target = np.sin(2*math.pi*X_train[:,0])

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
training_error = np.zeros(M_max)
testing_error  = np.zeros(M_max)
train_output = np.zeros((M_max,nt))
test_output  = np.zeros((M_max,ntest))
for i in range(M_max):
	train_output[i] = closed_line_fit_polynomial(train_samples,t_train,i,lambbda,nt)
	test_output[i]  = closed_line_fit_polynomial(test_samples,t_test,i,lambbda,ntest)

	training_error[i]  = E_rms(t_train , train_output[i])
	testing_error[i]   = E_rms(t_test , test_output[i])

print('training_error')
print(training_error)
print('testing_error')
print(testing_error)
M = 5
phi_train = np.ones((nt,M))
for i in range(nt):
	l = X_train[i,0]
	phi = np.zeros(M)
	for j in range(M):
		phi[j] = l**j
	phi_train[i] = phi[:]


w_train = np.matmul(inv(np.matmul(phi_train.transpose(),phi_train)+ lambbda * np.identity(M)) ,np.matmul(phi_train.transpose(), train_samples))

x=np.linspace(0,1,nt)
# print(X_train[0])
# print(train_samples)
y=w_train[0]*train_samples**0
for i in range(M-1):
	y+= w_train[i+1]*train_samples**(i+1)
# for i in range(M-1):
# 	y += w_train[M] * x**M


line1, = plt.plot(training_error, marker='o', label='training_error')
line2, = plt.plot(testing_error, marker='o', label='testing_error')

plt.legend()
plt.show()

# xe=np.linspace(0,M_max ,M_max)
# plt.figure()
# red_patch = mpatches.Patch(color='red', label='testing_error')
# plt.legend(handles=[red_patch])
# blue_patch = mpatches.Patch(color='blue', label='training')
# plt.legend(handles=[blue_patch])
# plt.plot(xe,training_error,'b.',xe,testing_error,'r.')
# plt.show()
#plotting
# plt.figure()
# plt.subplot(121)
# plt.plot(train_samples,t_train, 'go',x,y, 'm')
# plt.title('Closed Form')
# plt.ylabel('sin with noise')
# plt.xlabel('random x values from uniform distribution')
# plt.show()








