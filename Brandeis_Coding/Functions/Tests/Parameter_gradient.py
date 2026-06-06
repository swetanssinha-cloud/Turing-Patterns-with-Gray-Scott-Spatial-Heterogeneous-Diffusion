import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


k = 0.06 #(Stripes) #0.028, 0.064 (Dots) #0.010, 0.046 (Dynamic) #0.037, 0.06 (Standard)
N = 200
L = 200
dx = L / N
height = 5
dy = dx
n_steps =  200 #10000 #Using 300 for video. 
# p1 = 50. #normally what I use
# p2 = 150

p1 = 60
p2 = 200-p1
s = 0.1
x = np.linspace(0,L,N,endpoint=False)
print('x', x)
scale = 0.16



input1 = s * (x - p1) #so, this creates a bunch of values between the two boundries and creates them into inputs for the tanh functions
input2 =  s * -1 * (x - p2)
tanfunc =   0.37 * ((height - 1)/2 * (2-np.tanh(input1) - np.tanh(input2)) +1) #now that we have a bunch of values between these two boundries, we can use the tanh function to create a bunch of diffusion constants between those two boundries

Du=0.16
Dv=0.08

dt = 0.1

F = np.full((N,N), tanfunc) 

# print('Du max', Du.max())


# print('Du', Du)
# print('Dv', Dv)

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
    # should this not be (d2Z_dx2/dx * dx) + (d2Z_dy2/dy * dy)? I think we are saying that 
    #dy = dx. So it doesn't matter and we combine like terms. 
    return derivative

def solver(U, V, Du, Dv, F, k, dx, dt):
    Lu = laplacian(U, dx)
    Lv = laplacian(V, dx)

    uvv = U * V * V

    U += (Du * Lu  - uvv + F * (1 - U)) * dt
    V += (Dv * Lv  + uvv - (F+k) * V) * dt
    #Here is forward euler's method for the next time step. 
    return U, V
U = np.ones((N,N))
V = np.zeros((N,N))


r = 20
U[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.50 #orginally was 0.50
V[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.25 #orginally was 0.25
U += 0.05 * np.random.rand(N,N)
V += 0.05 * np.random.rand(N,N)

fig, ax = plt.subplots(figsize=(8,6)) #regular sizing in a cube

im = ax.imshow(V, cmap='inferno', interpolation='bilinear', vmin=0, vmax=0.5)
plt.axis('on')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('[V]')

tanh_y = (tanfunc - tanfunc.min()) / (tanfunc.max() - tanfunc.min()) * (N - 1)

# x-coordinates: pixel indices (0 to N-1)
x_coords = np.arange(N)

# Overlay line: static sine curve
tanh_line, = ax.plot(x_coords, tanh_y, color='red', linewidth=3, alpha=0.6)

# Set axes to avoid autoscaling that might hide the line
ax.set_xlim(0, N-1)
ax.set_ylim(0, N-1)

ax.set_title('F is plotted as a function of space')

def update(frame, U, V, Du, Dv, F, k, dx, dt, tanh_line):
    for _ in range(100):
        U, V = solver(U, V, Du, Dv, F, k, dx, dt)
    im.set_array(V)
    return [im, tanh_line]
ani = animation.FuncAnimation(fig, update, frames=n_steps, interval = 0.1, blit=True, fargs=(U, V, Du, Dv, F, k, dx, dt, tanh_line))
#ani.save('Gray_Scott_Animation.mp4', writer='ffmpeg', fps=30, dpi=300) #Save the animation as a video file
plt.show()

