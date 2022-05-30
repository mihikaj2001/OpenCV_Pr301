import matplotlib.pyplot as plt
import numpy

txt_file = open("orb_similarityVals.txt", "r")
rawdata = txt_file.read()

data = rawdata[1:-1]
print(data)

i_data = data.split(",")
print(i_data)

def similarityPlot(similarity_array):
  y = similarity_array
  n = len(y)
  # print(float(y[99]))
  Y=[]
  c=0
  for i in range(n):
    Y.append(float(y[i]))
    if(Y[i]<=0.2):
      c=c+1
  
  print(Y)
  print(c)
  x = []
  for i in range(n):
    x.append(i)
  # print(len(x))
  print(len(x))
  plt.plot(x,Y)
  plt.show()
  # plt.scatter(x,y)
  # plt.show()
  
similarityPlot(i_data)