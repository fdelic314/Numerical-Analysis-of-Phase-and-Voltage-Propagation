import numpy as np

# Import the neighbouring nodes
i, j = np.loadtxt("C:/Users/Fahrudin Delic/Desktop/nglines1.csv", skiprows=1 , unpack=True, delimiter=";")


# Obtain the adjecency matrix

A = np.zeros((int(np.max(np.concatenate([i,j]))),int(np.max(np.concatenate([i,j])))))

k = 0
while k < int(len(i)):
    i_pos = i.astype(int)[k] - 1
    j_pos = j.astype(int)[k] - 1
    A[i_pos, j_pos] = 1
    A[j_pos, i_pos] = 1
    k = k + 1
    

# Save the matrix 
B = np.save("C:/Users/Fahrudin Delic/Desktop/nglines_Adj_matrix.npy", A)
    
    
           