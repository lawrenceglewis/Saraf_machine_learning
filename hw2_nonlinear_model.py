import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv


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
print(X_train[:,0])
# target vector
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

# degree of polynomial
M = 4

# weight decay factor, reduces higher order terms for line fitting
lambbda  = 0
phi_train = np.ones((nt,M))
for i in range(nt):
	l = X_train[i,0]
	phi = np.zeros(M)
	for j in range(M):
		print (range(M))
		phi[j] = l**j
	phi_train[i] = phi[:]

w_train = np.matmul(inv(np.matmul(phi_train.transpose(),phi_train)+ lambbda * np.identity(M+1)) ,np.matmul(phi_train.transpose(), train_samples))

y = np.matmul(X_train,w_train)

# print(phi_train)
# print (test_samples)
# print(train_samples)



#plotting
plt.figure()
plt.subplot(121)
plt.plot(train_samples,t_train, 'go',w_train,y, 'm')
plt.title('Closed Form')
plt.ylabel('sin with noise')
plt.xlabel('random x values from uniform distribution')







#generate a uniform distribution from 0-10 with 100 samples
# a = 10
# u_distribution  = np.ones((a,a))
# multiplier = np.zeros((a,a))
# for a in range(a):
# 	multiplier[a,a] = a + 1


# u_distribution  = np.matmul(u_distribution,multiplier)
# u_dist_flat = u_distribution.flatten()
# samples = np.random.choice(np.squeeze(u_dist_flat),a+1)
# print(samples)
