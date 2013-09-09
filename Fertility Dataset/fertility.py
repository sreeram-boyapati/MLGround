#Find the weights for the data
#Write Sigmoid Function
#Write Cost Function
#Plot Graphs (Testing Data)
#Regularize the data
from scipy import optimize as opt
import pickle
import numpy as np
import math



def num(s):
	try:
		return int(s)
	except Exception, e:
		return float(s)

data = open("fertility_Diagnosis.txt")
output = open("output.txt",'w+')
clean = data.read().replace('N','1').replace('O','0')
output.write(clean)
clean = clean.split('\n')
Data_Matrix = []
for i in clean:
	data_row = i.split(',')
	for index, j in enumerate(data_row):
		if(j is not ''):
			data_row[index] = num(j)
	if(len(data_row) is 10):
		Data_Matrix.append(data_row)

def sigmoid(s,x):
	f = s.dot(x)
	Sig_Matrix = []
	for i in f:
		t = 1/(1 + math.exp(-i))
		Sig_Matrix.append(t)
	Sig_Matrix = np.array(Sig_Matrix)
	return Sig_Matrix

def sigmoid1(s):
	t = 1/(1+math.exp(-s))
	return t
Data_Matrix = np.array(Data_Matrix)
Input_Data = Data_Matrix[0:, 0:9]
Output_Data = Data_Matrix[0:, 9]
Theta_Matrix = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype='float')

#h(theta) has been initialized...p.zeros((1, 9), dtype = np.float) 
Sig_Matrix = sigmoid(Input_Data, Theta_Matrix)
def CostFunction(Smatrix, Omatrix):
	try:
		s1 = np.log10(Smatrix)
		s2 = np.log10(1 - Smatrix)
		e = - Omatrix * s1 - (1 - Omatrix) * s2 # -y.*log(h(theta)) - (1 - y).*log(h(theta))
		return e
	except Exception, e:
		raise e


def CostFunction_Wrapper(Theta_Matrix, Input_Data, Output_Data):
	Cost_Vector=[]
	size_of_data=len(Input_Data)
	for i in range(0, len(Theta_Matrix)):
		s = Input_Data[i].dot(Theta_Matrix)
		sig = sigmoid1(s)
		OD = Output_Data[i]
		cost_row = CostFunction(sig, OD)
		Cost_Vector.append(cost_row)
	#Calculating Derivative of Cost Function
	Cost_Gradient = []
	sum1 = 0
	#Calculating summation(h(theta) - y)
	for i in range(0, len(Theta_Matrix)):
		gradient_row = []
		sig = sigmoid1(Input_Data[i].dot(Theta_Matrix))
		h = sig - Output_Data[i]
		sum1 += h

	for j in Input_Data[i]:
			Cost_Gradient.append(sum1 * j)
	Cost_Gradient = np.array(Cost_Gradient)/size_of_data
	Cost_Vector = sum(np.array(Cost_Vector))/size_of_data
	return Cost_Vector

#[Cost_Vector, Cost_Gradient] = CostFunction_Wrapper(Theta_Matrix, Input_Data, Output_Data)
#Gradient Descent- Scipy Optimize function
Theta_Matrix = np.transpose(Theta_Matrix)
min_theta = opt.fmin(CostFunction_Wrapper, Theta_Matrix, args=(Input_Data, Output_Data))
print min_theta