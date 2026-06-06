import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


F, k = 0.028, 0.064 #0.046, 0.064 (Stripes) #0.028, 0.064 (Dots) #0.010, 0.046 (Dynamic) #0.037, 0.06 (Standard)
N = 200
L = 200
dx = L / N
dt = 0.1
n_steps =  300 #10000 #Using 300 for video. 

Dv = 2
Du = 1
Dv_array = np.full((N,N),Dv)
Dv_array[:,:25] = Dv * 2 #This is the new Dv value for the area of interest
Dv_array[:,75:] = Dv * 2
Du_array = np.full((N,N),Du)
Du_array[:,:25] = Du * 2 #This is the new Du value for the area of interest
Du_array[:,75:] = Du * 2 
def laplacian(Z, dx):
    d2Z_dx2 = np.roll(Z, 1, 0) - 2 * Z + np.roll(Z, -1, 0)
    d2Z_dy2 = np.roll(Z, 1, 1) - 2 * Z + np.roll(Z, -1, 1) 

    derivative = (d2Z_dx2 + d2Z_dy2)/(dx * dx)
    return derivative

# dDvx= (np.roll(Dv, 1, 0) - Dv)/dx
# dDvy = (np.roll(Dv, 1, 1) - Du)/dx
    
# dDux = (np.roll(Du, 1, 0) - Du)/dx
# dDuy = (np.roll(Du, 1, 1) - Du)/dx

def solver(U, V, Du_array, Dv_array, F, k, dx, dt):
    Lu = laplacian(U, dx) 
    Lv = laplacian(V, dx) 
    uvv = U * V * V

    U += ( Du_array * Lu - uvv + F * (1 - U)) * dt
    V += ( Dv_array * Lv + uvv - (F+k) * V) * dt
    #Here is forward euler's method for the next time step. 
    return U, V
U = np.ones((N,N))
V = np.zeros((N,N))

r = 20
U[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.50
V[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.25
U += 0.05 * np.random.rand(N,N)
V += 0.05 * np.random.rand(N,N)

fig, ax = plt.subplots(figsize=(8,6)) #regular sizing in a cube

im = ax.imshow(V, cmap='inferno', interpolation='bilinear', vmin=0, vmax=1)
plt.axis('off')



def update(frame, U, V, Du_array, Dv_array, F, k, dx, dt):
    for _ in range(100):

        U, V = solver(U, V, Du_array, Dv_array, F, k, dx, dt)
    im.set_array(V)
    return [im]
ani = animation.FuncAnimation(fig, update, frames=n_steps, interval = 0.1, blit=True, fargs=(U, V, Du_array, Dv_array, F, k, dx, dt))
#ani.save('Gray_Scott_Animation.mp4', writer='ffmpeg', fps=30, dpi=300) #Save the animation as a video file
plt.show()
# Ok so this lowk works, just oddly. There is a section on in the middle but adjusted to the left where the diffusion is HELLA slow. So its
#really weird. This kinda works, but there is DEF a better way to do this. So a lesson is that you also have to make Du a matrix
#then you can apply that section of the matrix to the Lu and Lv calcs. It is a bit weird, but it works.
#So the idea is that you can change the diffusion rate in a specific area of the grid
#and it will change the diffusion rate in that area. So you can have a high diffusion rate in one area and a low diffusion rate in another area.
#This is useful for simulating different diffusion rates in different areas of the grid.
#So you can have a high diffusion rate in one area and a low diffusion rate in another area.


#NOTE - I just did the same thing with Dv and got a new result. I doubled it. 