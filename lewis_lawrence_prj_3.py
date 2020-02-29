import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.legend_handler import HandlerLine2D
from numpy.linalg import inv

L = 100
N = 25
sin_lowerbound = 0
sin_upperbound = 1
standard_deviation_basis_gause = .1
seed = 0

# noise from gausian distribution with mean = 0 standard deviation = .3
mu, sigma = 0, .3

#try m = 1
m_max, lambd= 1, .3

#what to step rbf by
rbf_step = .1

rbf_itteration = int(m_max/rbf_step)

#create X values from uniform distribution. Get target values from sin function plus error.
def sample_sin_and_make_target(seed_number,sin_lower_bound,sin_upper_bound,number_of_samples,error_mean,error_standard_deviation):

	# set seed so that we have same distribution every time
	np.random.seed(seed_number)

	# sample nt samples from a uniform distribution
	uniform_samples = np.random.uniform(sin_lower_bound,sin_upper_bound,number_of_samples)
	
	e = np.random.normal(error_mean, error_standard_deviation, number_of_samples)

	# target vector with error
	target_vector = np.sin(2*math.pi*uniform_samples) + e
	return uniform_samples, target_vector

def create_gausan_vector_from_samples(samples_vector, standard_dev,mean):
	length = len(samples_vector)
	gausian_vector = np.zeros(length)
	for i in range(length):
		gausian_vector[i] = math.exp(-1/(2*math.pow(standard_dev,2))*math.pow((samples_vector[i]-mean),2))
	return gausian_vector

#rbf M/M_itterations must have no remainder
def closed_line_fit_gausian_rbf(input_data,target_data,M_max,lmbda,num_samp, std, M_step):
	dimensions = input_data.shape
	rows = dimensions[0]
	M = int(M_max/M_step)
	phi_train = np.ones((rows,M+1))

	# todo make it accept m columns for different dimensions
	# iterates for each sample
	for i in range(0,M+1):
		print(i)
		if i > 0:
			# print('input data size')
			# print(np.squeeze(input_data.shape))

			#print('gausian of input data')
			#print(np.squeeze(create_gausan_vector_from_samples(input_data[:,i-1], std,M_itteration)))

			phi_train[:,i] = create_gausan_vector_from_samples(input_data, std,i*M_step).transpose()
	#w_train = np.matmul(np.matmul(inv(np.matmul(phi_train.transpose(),phi_train)) ,phi_train.transpose()), input_data.transpose())#/*+ lmbda * np.identity(M+1)*/
	#w_train = np.matmul(np.linalg.pinv(phi_train),target_data)
	print('size of phi_train')
	print(phi_train.shape)
	print('size of target')
	print(target_data.shape)
	w_train = np.matmul(np.matmul(inv(np.matmul(phi_train.transpose(),phi_train)+lambd * np.identity(M+1)),phi_train.transpose()),np.squeeze(target_data))
	return w_train

def create_data_set(number_of_sets,number_of_samples_per_set):

	return data_set




# create 100 distributions
# store each distribution in a column
sample_distributions = np.zeros((N,L))
target_distributions = np.zeros((N,L))
for i in range (L):
	sample_distributions[:,i],target_distributions[:,i] = sample_sin_and_make_target(i,sin_lowerbound,sin_upperbound,N,mu,sigma)

weights = closed_line_fit_gausian_rbf(sample_distributions[:,0].transpose(),target_distributions[:,0].transpose(),m_max,lambd,N, standard_deviation_basis_gause,rbf_step)



print('weights')
print(weights)




# create ideal sin
ideal_x = np.linspace(0,1,N*10)
ideal_y = np.sin(2*math.pi*ideal_x)

# for each distribution line fit with gausian model
#for i in range(L):
#plotting
plt.figure()
plt.plot(ideal_x,ideal_y,'b')
#print(sample_distributions)
for i in range(L-1):
	plt.plot(sample_distributions[:,i],target_distributions[:,i],'r.')
plt.show()
