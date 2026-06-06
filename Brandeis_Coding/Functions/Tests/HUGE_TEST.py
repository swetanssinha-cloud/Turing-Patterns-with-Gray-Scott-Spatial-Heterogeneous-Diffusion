
from IPython.display import HTML
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

F, k =  0.046, 0.064 #(Stripes) #0.028, 0.064 (Dots) #0.010, 0.046 (Dynamic) #0.037, 0.06 (Standard)
N = 200
L = 200
dx = L / N

dy = dx
n_steps = 400 #10000 #Using 300 for video. 
sharpness = 30 #this will control how sharp the chnage in concentration will be at the boundry p1 and p2
scale = 0.16


Du = np.zeros((N,N))

x = np.linspace(0,L,N)
y = np.linspace(0,L,N)

X,Y = np.meshgrid(x,y)


##################FUNCTION START

s=3 #slope - rotates about a point
o = 33 #shift on x-axis
l = 0 # Some shift in a weird direction - keep constant
k = 57 #width control - in order to change this you must graph on desmos first and then see how to change l and u repsectivly
u = 2.5 #Another way to control width - just changes in one direction tho - better to change l and shift. u should not be too big or the graph will not be on this coordinate size. 
maxDu = 1.5 #maximum Du

cat = (s * (X-o) + (Y-l) - k)/sharpness
brill = -1 * (s * (X-o) + (Y - l) - u * k)/sharpness
silverman = (-s * (X-o) + (Y-l) - k)/sharpness
brooks = -1 * (-s * (X-o) + (Y-l) - u * k)/ sharpness

tanh_top_new = ((maxDu - 1)/2) * (-np.tanh(cat)-np.tanh(brill)+2) + 1
tanh_bottom_new = ((maxDu -1)/2 * (-np.tanh(silverman)-np.tanh(brooks)+2))+1

################FUNCTION DONE

Du[0:N//2, :] = scale * tanh_top_new[0:N//2, :]
Du[N//2:N, :] = scale * tanh_bottom_new[N//2:N, :]

     



Dv =  (np.min(Du)/2) * np.ones_like(Du)  # Dv = 0.08 For now I have placed Dv = Du/2 because that is the ratio we have been using so far. I still don't know why works, but it does.

dt = ((dx)**2)/(4 * Du.max()) * 0.05 #This is the time step, which is based on the diffusion constant. This is a function for the time step. 


def first_derivative(M, axis):
    firstD = (np.roll(M, 1, axis) - np.roll(M, -1, axis)) / (2 * dx)
    return firstD
        
def combined_derivative(D, Concentration):
    dD_dx = first_derivative(D, 0)
    dU_dx = first_derivative(Concentration, 0)
    dD_dy = first_derivative(D, 1)
    dU_dy = first_derivative(Concentration, 1)
    return dD_dx * dU_dx + dD_dy * dU_dy


def laplacian(Z, dx):
    d2Z_dx2 = np.roll(Z, 1, 0) - 2 * Z + np.roll(Z, -1, 0)
    d2Z_dy2 = np.roll(Z, 1, 1) - 2 * Z + np.roll(Z, -1, 1) 

    derivative = (d2Z_dx2 + d2Z_dy2)/(dx * dx) 
    
    return derivative

def solver(U, V, Du, Dv, F, k, dx, dt):
    Lu = laplacian(U, dx)
    Lv = laplacian(V, dx)

    uvv = U * V * V

    U += (Du * Lu + combined_derivative(Du,U) - uvv + F * (1 - U)) * dt
    V += (Dv * Lv + combined_derivative(Dv,V) + uvv - (F+k) * V) * dt
    #Here is forward euler's method for the next time step. 
    return U, V
U = np.ones((N,N))
V = np.zeros((N,N))


r = 20
U[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.75 #orginally was 0.50
V[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.50 #orginally was 0.25
U += 0.05 * np.random.rand(N,N)
V += 0.05 * np.random.rand(N,N)

# fig, ax = plt.subplots(figsize=(8,6)) #regular sizing in a cube

# im = ax.imshow(V, cmap='inferno', interpolation='bilinear', vmin=0, vmax=1)
# plt.axis('on')
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label('[V]')

total_V_concentration = np.zeros(n_steps)
total_time = np.zeros(n_steps)

# def update(frame, U, V, Du, Dv, F, k, dx, dt):
#     for _ in range(100):
#         U, V = solver(U, V, Du, Dv, F, k, dx, dt)
    
#     total_time[frame] = frame * dt * 100
#     total_V_concentration[frame] = np.mean(V)


#     im.set_array(V)
#     return [im] 
  
for i in range(n_steps):
    for _ in range(100):
        U, V = solver(U, V, Du, Dv, F, k, dx, dt)
    
    total_time[i] = i * dt 
    total_V_concentration[i] = np.mean(V)


# im.set_array(V)
   

# matplotlib.rcParams['animation.embed_limit'] = 1000


# ani = animation.FuncAnimation(fig, update, frames=n_steps, interval = 0.1, blit=False, fargs=(U, V, Du, Dv, F, k, dx, dt))
# plt.show()
# HTML(ani.to_jshtml())



plt.figure(figsize=(6,5))
plt.imshow(V, cmap='viridis')
plt.title('Gray Scott')
plt.colorbar(label='V')


plt.show()



