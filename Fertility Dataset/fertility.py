#Find the weights for the data
#Write Sigmoid Function
#Write Cost Function
#Plot Graphs (Testing Data)
#Regularize the data

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
	data_row=i.split(',')
	for index, j in enumerate(data_row):
		if(j is not ''):
			data_row[index] = num(j)
	if(len(data_row) is 10):
		Data_Matrix.append(data_row)

def sigmoid(s,x):
	t = 1/(1+math.exp(-s*x))
	return t

def sigmoid1(s):
	t = 1/(1+math.exp(-s))
	return t
Data_Matrix = np.array(Data_Matrix)
Input_Data = Data_Matrix[0:, 0:9]
Output_Data = Data_Matrix[0:, 9]
Theta_Matrix = np.zeros((1, 9), dtype = np.float)
sigmoid_convert = np.vectorize(sigmoid)

#h(theta) has been initialized... 
Sig_Matrix = sigmoid_convert(Input_Data, Theta_Matrix)

def CostFunction(Smatrix, Omatrix):
	try:
		s1=np.log10(Smatrix)
		s2=np.log10(1 - Smatrix)
		e = - Omatrix * s1 - (1 - Omatrix) * s2
		return e
	except Exception, e:
		raise e

#def CostFunction(Smatrix, Omatrix):
	try:
		e = - Omatrix * math.log10(Smatrix) - (1 - Omatrix) * math.log10(1 - Smatrix)
		return e
	except Exception, e:
		raise e
cost_convert = np.vectorize(CostFunction)

def CostFunction_Wrapper(Input_Data,Output_Data,Theta_Matrix):
	Cost_Vector=[]
	size_of_data=len(Input_Data)
	for i in range(0, len(Theta_Matrix)):
		s=Input_Data[i].dot(np.transpose(Theta_Matrix))
		sig=sigmoid1(s)
		OD=Output_Data[i]
		cost_row =CostFunction(sig, OD)
		Cost_Vector.append(cost_row)
	#Calculating Derivative of Cost Function
	Cost_Gradient=[]
	sum1=0
	#Calculating summation(h(theta) - y)
	for i in range(0, len(Theta_Matrix)):
		gradient_row=[]
		sig = sigmoid1(Input_Data[i].dot(np.transpose(Theta_Matrix)))
		h = sig - Output_Data[i]
		sum1 += h

	for j in Theta_Matrix[i]:
			Cost_Gradient.append(sum1*j)
	Cost_Gradient=np.array(Cost_Gradient)/size_of_data

	Cost_Vector=sum(np.array(Cost_Vector))/size_of_data
	return [Cost_Vector, Cost_Gradient]

[Cost_Vector, Cost_Gradient]=CostFunction_Wrapper(Input_Data,Output_Data,Theta_Matrix)
print Cost_Gradient
#Gradient Descent