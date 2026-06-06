import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation    

def laplacian(Z, dx):
    d2Z_dx2 = np.roll(Z, 1, 0) - 2 * Z + np.roll(Z, -1, 0)
    d2Z_dy2 = np.roll(Z, 1, 1) - 2 * Z + np.roll(Z, -1, 1)
    derivative = (d2Z_dx2 + d2Z_dy2)/(dx * dx)
    return derivative

def solver(U, V, Du, Dv, F, k, dx, dt):
    Lu = laplacian(U, dx)
    Lv = laplacian(V, dx)
    uvv = U * V * V

    U += (Du * Lu - uvv + F * (1 - U)) * dt
    V += (Dv * Lv + uvv - (F+k) * V) * dt
    return U, V

if __name__ == "__main__":

    Du, Dv = 0.16, 0.08
    N = 200
    L = 200
    dx = L / N
    dt = 1
    n_steps = 10000

    # F = np.linspace(0.020,0.050,6)
    # k = np.linspace(0.040,0.070,6) #orginal - this is zoomed in. 

    F = np.linspace(0.01,0.10,6)
    k = np.linspace(0.01, 0.10,6)  #This will be for non zoomed in (intital phase chart)


    for i in range(len(F)):
        for j in range(len(k)):

            U = np.ones((N,N))
            V = np.zeros((N,N))

            r = 20
            U[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.50
            V[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.25
            U += 0.05 * np.random.rand(N,N)
            V += 0.05 * np.random.rand(N,N)

            fig, ax = plt.subplots()
            im = ax.imshow(V, cmap='viridis', interpolation='bilinear', vmin=0, vmax=0.5)
            plt.axis('off')
            #addtion of color bar for [v]
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('[V]') #NEW ADDTION

            # for i in range(n_steps): 
            #     for _ in range(100):
            #         U, V = solver(U, V, Du, Dv, F, k, dx, dt)
                

            def update(frame, U, V, Du, Dv, F, k, dx, dt):
                for _ in range(100):
                    U, V = solver(U, V, Du, Dv, F, k, dx, dt)
                im.set_array(V)
                return [im]

            ani = animation.FuncAnimation(fig, update,
                                          frames=n_steps,
                                          interval=0.1,
                                          blit=True,fargs=(U, V, Du, Dv, F[i], k[j], dx, dt))
            plt.show()
            fig.tight_layout()  # Adjusts layout so color bar fits
            # fig.savefig(f"LFT_F{F[i]:.3f}_k{k[j]:.3f}.png", bbox_inches='tight')#T = test and LF = last frame
            plt.imsave(f"LF_F{F[i]:.3f}_k{k[j]:.3f}.png", V, cmap='viridis', vmin=0, vmax=0.5) #last frame =


