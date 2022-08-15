import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings("ignore")
#Populating the interaxtive namespace from numpy and matplotlib
import networkx as nx

def grid_var(Adj_matrix, x_P):
    G = nx.from_numpy_matrix(Adj_matrix)
    p = nx.shortest_path(G)
    g_dist = np.zeros(np.shape(Adj_matrix)[0])
    
    i = 0
    while i < (np.shape(Adj_matrix)[0]):
        j = 0
        while j < len(p[x_P][i]):
            g_dist[p[x_P][i][j]] = j
            j = j + 1
            
        i = i + 1
    return g_dist**2


# Upload adjecency matrix

A = np.load(r"C:/Users/Fahrudin Delic/Desktop/nglines_Adj_matrix.npy")


# Upload the simulation data
V = np.load("Voltage nig.npy")
f = np.load("Frequency nig.npy")

# Step size
dt = 0.001 

# Get the shape and the time arrays
size = np.shape(V)
tarray = np.linspace(0, int(size[0] * dt), size[0])
x_P = 23 # Starting position

fig1 = plt.figure(figsize=(7,5))
legend_properties = {'weight':'bold'}
plt.plot(tarray,V[:, int(x_P)],tarray,V[:, int(3)],tarray,V[:, int(69)])
plt.ylabel(" Voltage deviation, $\delta V_i$ (p.u.)", fontsize = 15,fontweight='bold')
plt.xlabel("$t - t_0$ (s)", fontsize = 15,fontweight='bold')
plt.legend(["24 [$dist=0$]","4 [$dist=4$]","70 [$dist=8$]"],prop=legend_properties)
plt.xticks(fontweight='bold',fontsize=12)
plt.yticks(fontweight='bold',fontsize=12)
#fig1.savefig('Voltage displacement nig1.png', dpi=150, bbox_inches= 'tight')

# Obtain analytical solution of decay
max_dev = np.max(f[:,int(x_P)])
f_anal = max_dev * np.exp(-0.1 * tarray)

fig2 = plt.figure(figsize=(7,5))
legend_properties = {'weight':'bold'}
plt.plot(tarray,f[:, int(x_P)],tarray,f[:, int(3)],tarray,f[:, int(69)], tarray, f_anal)
plt.ylabel("Frequency deviation, $w_i$ (p.u.)", fontsize = 15,fontweight='bold')
plt.xlabel("$t - t_0$ (s)", fontsize = 15,fontweight='bold')
plt.legend(["24 [$dist=0$]","4 [$dist=4$]","70 [$dist=8$]", "analytical"],prop=legend_properties)
plt.xticks(fontweight='bold',fontsize=12)
plt.yticks(fontweight='bold',fontsize=12)
#fig2.savefig('Frequency displacement nig1.png', dpi=150, bbox_inches= 'tight')

# Log log plot
# Analytical function
a = np.array(tarray)
b = 0.000122 * np.exp(-0.1 * a)

# Slicing of data

t_ramp = 0.3
slice_start = int(t_ramp / dt)


fig3 = plt.figure(figsize=(7,5))
legend_properties = {'weight':'bold'}
plt.loglog(tarray[slice_start:],V[slice_start:, int(x_P)],tarray[slice_start:],b[slice_start:])
plt.ylabel("Log voltage deviation, , $\delta V_i$", fontsize = 15,fontweight='bold')
plt.xlabel("Log $t - t_0$ ", fontsize = 15,fontweight='bold')
plt.legend(['Simulated data','$exp ( - \Gamma \cdot t )$'],prop=legend_properties)
plt.xticks(fontweight='bold',fontsize=12)
plt.yticks(fontweight='bold',fontsize=12)
#fig3.savefig('Loglog voltage nig.png', dpi=150, bbox_inches= 'tight')

# Finding the variances
q = 0
qmax = size[0]
qarray = []
var_array_V = []
var_array_f = []

var_coef = grid_var(A, x_P)

while q < qmax:
     # Frequency variance
    var_f_n = np.sum(var_coef * np.abs(f[q,:]))/(np.sum(np.abs(f[q,:])))
    var_array_f += [float(var_f_n)] # Update frequency varience array
    # Voltage variance
    var_V_n = np.sum((var_coef * np.abs(V[q,:])))/(np.sum(np.abs(V[q,:])))
    var_array_V += [float(var_V_n)] # Update voltage varience array
    qarray += [q]
    q += 1
    
    
fig4 = plt.figure(figsize=(7,5))
legend_properties = {'weight':'bold'}
plt.plot(tarray,var_array_V)
plt.ylabel("$<(x_i - x_0)^2_{\delta V_i}>$", fontsize = 15,fontweight='bold')
plt.xlabel("$t - t_0$ (s)", fontsize = 15,fontweight='bold')
#plt.legend(['$dist=0$','$dist=5$','$dist=20$'],prop=legend_properties)
plt.xticks(fontweight='bold',fontsize=12)
plt.yticks(fontweight='bold',fontsize=12)
#fig4.savefig('Voltage variance nig.png', dpi=150, bbox_inches= 'tight')

fig5 = plt.figure(figsize=(7,5))
legend_properties = {'weight':'bold'}
plt.plot(tarray,var_array_f)
plt.ylabel("$<(x_i - x_0)^2_{\delta \omega_i}>$", fontsize = 15,fontweight='bold')
plt.xlabel("$t - t_0$ (s)", fontsize = 15,fontweight='bold')
#plt.legend(['$dist=0$','$dist=5$','$dist=20$'],prop=legend_properties)
plt.xticks(fontweight='bold',fontsize=12)
plt.yticks(fontweight='bold',fontsize=12)
#fig5.savefig('Frequency variance nig.png', dpi=150, bbox_inches= 'tight')

# Plot the heat maps

# Rearrange the arrays with geometric distance
geom_dist = grid_var(A, x_P)**(1/2) # Get the geodesic distance
sort_array = np.argsort(geom_dist) # Get the indicies for sorting

# Create sorting arrays
x_len = np.shape(f)[1]
y_len = np.shape(f)[0]

y_sort = np.zeros((y_len,x_len))
yi_sort = np.zeros((y_len,x_len))
V_sort = np.zeros((y_len,x_len))
d_sort = np.zeros(x_len)

# Sort the arrays

i1 = 0
while i1 < x_len:
    yi_sort[:,i1] = f[:,sort_array[i1]]
    V_sort[:,i1] = V[:, sort_array[i1]]
    d_sort[i1] = int(geom_dist[sort_array[i1]])
    i1 = i1 + 1
    
# Create the spatial coordinates of Nigerian grid
x = np.linspace(0,int(np.shape(A)[0] - 1), int(np.shape(A)[0]))

tp = 3 # Plotting seconds
tpq = int(tp/dt - 1)
tarray1 = np.round(np.array(tarray[0::tpq]),2)

x_sample = np.array([x[1], x[8], x[21], x[34], x[45], x[55], x[62], x[67]])
d_sample = np.array([d_sort[1], d_sort[8], d_sort[21], d_sort[34], d_sort[45], d_sort[55], d_sort[62], d_sort[67]])


fig12 = plt.figure(figsize=(7,5))
plt.imshow(V_sort, aspect = "auto", cmap='viridis')
plt.xticks(x_sample, [1,2,3,4,5,6,7,8], fontweight='bold',fontsize=12)
plt.yticks(qarray[0::tpq], [0,3.0,6.0,9.0,12.0,15.0], fontweight='bold',fontsize=12)
plt.ylabel("$t - t_0$ (s)", fontsize = 15,fontweight='bold')
plt.xlabel("Geodesic distance", fontsize = 15,fontweight='bold')
plt.colorbar()
plt.grid(None)
#fig12.savefig('Voltage petrubation heat map nig', dpi=150, bbox_inches= 'tight')

fig13 = plt.figure(figsize=(7,5))
plt.imshow(f, aspect = "auto", cmap='viridis')
#plt.xticks(x[0::6], distance_sort[0::6])
plt.xticks(x_sample, [1,2,3,4,5,6,7,8], fontweight='bold',fontsize=12)
plt.yticks(qarray[0::tpq], [0,3.0,6.0,9.0,12.0,15.0], fontweight='bold',fontsize=12)
plt.ylabel("$t - t_0$ (s)", fontsize = 15,fontweight='bold')
plt.xlabel("Geodesic distance", fontsize = 15,fontweight='bold')
plt.colorbar()
plt.grid(None)
#fig13.savefig('Frequency petrubation heat map nig', dpi=150, bbox_inches= 'tight')