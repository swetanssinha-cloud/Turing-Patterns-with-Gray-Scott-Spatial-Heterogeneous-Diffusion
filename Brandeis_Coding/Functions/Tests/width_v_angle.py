import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

'''DERVATIVES'''
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

    U += (Du * Lu + combined_derivative(Du,U) - uvv + F * (1 - U)) * dt
    V += (Dv * Lv + combined_derivative(Dv,V) + uvv - (F+k) * V) * dt
    #Here is forward euler's method for the next time step. 
    return U, V

'''PARAMETERS'''
F, k =  0.046, 0.064 #(Stripes) #0.028, 0.064 (Dots) #0.010, 0.046 (Dynamic) #0.037, 0.06 (Standard)
N = 200
L = 200
dx = L / N
dy = dx
n_steps =  200 #10000 #Using 300 for video. 
x = np.linspace(0,L,N,endpoint=False)
scale = 0.16

'''FUNCTION INPUTS'''
H = 5 #HEIGHT

n = 12 #number of widths I have 

p = np.linspace(40,95,n) #First change in concavity
#second change in concavity
print(p)

s = 10 # Will change - sharpness of turn

angles = np.zeros(n)
st_d = np.zeros(n)

for j in range(len(p)):
    
    q = 200 - p 
    input_1 = (x-p[j])/s
    input_2 = (x-q[j])/s


    '''FUNCTION'''
    tanh = ((H - 1) /2) * (2-np.tanh(input_1)-np.tanh(input_2)) + 1

    '''APPLICATION OF FUNCTION TO Du'''
    Du = np.full((N,N), scale * tanh)
    Dv = (scale/2) * np.ones_like(Du)

    dt = ((dx**2)/(4 * Du.max())) * 0.9


    '''INTIAL PULSE'''

    U = np.ones((N,N)) #Intially set U and V to 0
    V = np.zeros((N,N))
    r = 20


    U[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.75 #orginally was 0.50 #Give a pulse to some certain area. 
    V[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.50 #orginally was 0.25

    U += 0.05 * np.random.rand(N,N) #Some small noise - just so things start happening a bit quicker
    V += 0.05 * np.random.rand(N,N)


    '''CALCULATIONS (FOR LOOP)'''
    for i in range(n_steps):
        for _ in range(100):
            U,V = solver(U, V, Du, Dv, F, k, dx, dt)


    '''VECTOR CALCS'''

    dVx = first_derivative(V,1)
    dVy = first_derivative(V,0)

    # '''MASKS'''
    # x2, y2 = np.meshgrid(np.arange(N), np.arange(N))

    # mask1 = x2 < p[j]
    # mask2 = x2 > q[j]


    # final_mask = mask1 | mask2

    # dVx =~final_mask * dVx
    # dVy= ~final_mask * dVy

    '''ANGLE CALCS'''

    theta = np.arctan2(dVy, dVx)

    theta[theta < 0] += np.pi

    theta = (theta* 180)/np.pi

    mean_angle = np.mean(theta)

    st_d[j] = np.std(theta)

    angles[j] = mean_angle

'''PLOTTING WIDTH V ANGLE'''
width = q - p
print("plotting...")
plt.plot(width, angles)
plt.xlabel("Width")
plt.ylabel("Mean Angle (Degrees)")
plt.title("Mean Angle as a function of Width")
plt.show()
