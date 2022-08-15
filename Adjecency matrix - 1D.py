import numpy as np

k = 100 # Grid size

# Obtain the adjecency matrix
B = np.zeros((k,k))

i = 1
while i < (k-1):
    B[i,i+1] = B[i,i-1] = 1
    i = i + 1
    
B[0,1] = B[-1,-2] = 1

np.save("Test matrix", B)