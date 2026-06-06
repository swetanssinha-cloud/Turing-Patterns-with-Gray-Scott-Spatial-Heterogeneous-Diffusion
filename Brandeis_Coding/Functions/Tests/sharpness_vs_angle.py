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

    U += (Du * Lu  - uvv + F * (1 - U)) * dt
    V += (Dv * Lv  + uvv - (F+k) * V) * dt
    #Here is forward euler's method for the next time step. 
    return U, V

'''PARAMETERS'''
F, k =  0.046, 0.064 #(Stripes) #0.028, 0.064 (Dots) #0.010, 0.046 (Dynamic) #0.037, 0.06 (Standard)
N = 200
L = 200
p1 = 60
p2 = 200 - p1
dx = L / N
dy = dx
n_steps =  100 #10000 #Using 300 for video. 
x = np.linspace(0,L,N,endpoint=False)
scale = 0.16

s_lower = 0
s_upper = 0.5

'''Sharpness versus Theta'''

iteration_times = 10 # Normally going to be 5 - but I want to push my computational limits to the max :)

s = np.linspace(s_lower, s_upper, iteration_times)  # Will change - sharpness of turn - notice that sharpness basically is the same when values above 1 - so just trying this range

angles = np.zeros(iteration_times)

for j in range(len(s)):
    
    input_1 = (x-p1)*s[j]
    input_2 = (x-p2)*s[j]
    
    '''FUNCTION'''
    tanh = ((N - 1) /2) * (2-np.tanh(input_1)-np.tanh(input_2)) + 1

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

    bound = 20

    mask_edge1 = x < p2 - bound
    mask_edge2 = x > p2 + bound

    mask_edge = mask_edge1 | mask_edge2

    final_mask_edge = mask_edge1 | mask_edge2

    dVx2 =~final_mask_edge * dVx
    dVy2= ~final_mask_edge * dVy


    '''THETA CALCS'''

    a = [1,0]

    dV = [dVx2,dVy2]
    dV_new = np.stack((dVy, -1 * dVx), axis = 0)        
    def dot_product(a, b):
        return a[0] * b[0] + a[1] * b[1]

    resultant = dot_product(a, dV_new)

    cos_theta = resultant / (np.sqrt(dVx**2 + dVy**2) * np.sqrt(a[0]**2 + a[1]**2))

    cos_theta_squared = cos_theta**2

    cos_2_theta = 2 * cos_theta_squared - 1

    theta = 0.5 * np.arccos(cos_2_theta)

    theta = theta * (180/np.pi) #convert to degrees

    theta_region = theta[~final_mask_edge]

    angles[j] = np.mean(theta_region)
    

print('angles:', angles)


'''PLOTTING WIDTH V ANGLE'''

plt.plot(s, angles)
plt.xlim(s_lower, s_upper)
plt.ylim(0, 90)
plt.xlabel("sharpness")
plt.ylabel("Mean Angle (Degrees)")
plt.title("Mean Angle as a function of sharpness")
plt.show()