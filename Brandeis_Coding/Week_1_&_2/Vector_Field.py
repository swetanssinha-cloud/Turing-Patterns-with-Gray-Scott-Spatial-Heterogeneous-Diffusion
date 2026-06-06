import numpy as np
import matplotlib.pyplot as plt


# F = 0.032
# k = 0.058 #Labryinth

F = 0.028
k = 0.064

scale = 5 #for easier visualization
n = 20

window = 1
u = np.linspace(0, window, n, endpoint=False) #creating 5 equally spaced points on the graph for our x axis
v = np.linspace(0, window, n, endpoint=False) #creates 5 equally spaced points on the graph for our y axis
U, V = np.meshgrid(u, v) #This takes the points and creates a 2D array. So we have for u  = [0, 0.375, 0.75, 1.125, 1.5] and for v = [0, 0.375, 0.75, 1.125, 1.5] But now you take u and have 5 of those 
#arrays and then you do the same for v. Now you have an array considiting of two arrays. So each point in the array will have a combination of u and v. Each one will be a different combination of u and v
#Then what you would do is take those values and assign them to U and V. So U and V will go through each point in this odd new array and assign themselves to values. 
#Now in the next line you will take those values and calculate the velocities of those points from the equations below

dU = (-1 * U * V**2 + F *  (1-U))
dV = (U * V**2 - (F + k) * V)


magnitude = np.sqrt(dU**2 + dV**2)
dU /= magnitude
dV /= magnitude


u_null1 = (F)/(F+v **2)
u_null2 = (F + k)/(v)

# if np.any((dU==0) & (dV==0)):
#     plt.plot(u, v, 'ro', linewidth=5) #This will plot the points where the velocities are zero, which means that they are s

#So now you have velocities for certain points of u and v. So in a graph, if you were to take one axis as u and the other as v, you will have veolcoties for each point in the graph. 

plt.figure()

plt.xlim(0,window)
plt.ylim(0,window)
plt.xlabel('U')
plt.ylabel('V')
plt.title('Vector Field of Gray-Scott Model')
plt.grid()
plt.quiver(u, v, dU, dV, color='blue', scale=30) #This will now plot what is actually going on in the graph. 
plt.plot(v, u_null1, 'k--', label='U Nullcline') #This will plot the nullcline for U
plt.plot(v, u_null2, 'k:', label='V Nullcline') #This will plot the nullcline for V
plt.legend()


plt.savefig("Nullclines_LF_F{F[i]:.3f}_k{k[j]:.3f}.png")


plt.show() #This will show the graph

