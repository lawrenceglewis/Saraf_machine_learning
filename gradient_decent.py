
# Reading an excel file using Python 
import xlrd 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Give the location of the file 
loc = (r"C:\Users\Lawrence\machine learning\proj1Dataset.xlsx") 

# To open Workbook 
wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0) 

# Extracting number of rows 
excelRows = int(sheet.nrows)

# Extracting number of columns 
excelCols = int(sheet.ncols)

# design matrix
X = np.ones((excelRows -1, excelCols))

for i in range (excelRows -1):
	for j in range (excelCols - 1):
		#print(int(sheet.cell_value(i+1,j)))
		X[i,j] = int(sheet.cell_value(i+1,j))
weight = X[:,0]
number_of_deletes = 0;
#target vector is horsepower
horsepower = np.zeros((excelRows-1,1))
for i in range(excelRows - 1):
	if isinstance(sheet.cell_value(i+1, 1), str):
		excelRows -= 1
		#delete row if no value is found for horsepower
		X = np.delete(X,i-number_of_deletes,axis=0)
		weight = np.delete(weight,i-number_of_deletes,axis=0)
		horsepower = np.delete(horsepower,i-number_of_deletes,axis=0)
		number_of_deletes += 1
	else:
		horsepower[i-number_of_deletes,0] = int(sheet.cell_value(i+1, 1))

X_T = X.transpose()

# solving weight vector with the closed form
wv = inv(X.transpose().dot(X)).dot(X.transpose()).dot(horsepower)

#output 
y = np.matmul(X,wv)
#------------------------Start Gradient Decent-----------------------
#number of itterations
k = 1000
#learning rate
l = 10**-12
#demensions to 1 demension
D = 1
#weight vectors
w = np.ones(D+1)

for i in range(k):
	# new weights = old weights - learning rate * gradient
	w = np.squeeze(np.asarray(np.subtract(w,(l*((2 * w.transpose().dot(X_T).dot(X) )- (2 * horsepower.transpose().dot(X)))))))

x=np.linspace(1500,5500,3500)
print(w)
yg = w[0]*x+w[1]

#plotting
plt.figure()
plt.subplot(121)
plt.plot(weight,horsepower, 'go',weight,y, 'm')
plt.title('Closed Form')
plt.ylabel('horsepower')
plt.xlabel('weight')

plt.subplot(122)
plt.plot(weight,horsepower, 'b.' )
plt.plot(x,yg,'r')
plt.title('Gradient Decent')
plt.ylabel('horsepower')
plt.xlabel('weight')
plt.show()



