import pickle
data=open("fertility_Diagnosis.txt")
output=open("output.txt",'w+')
clean=data.read().replace('N','1').replace('O','0')
output.write(clean)
