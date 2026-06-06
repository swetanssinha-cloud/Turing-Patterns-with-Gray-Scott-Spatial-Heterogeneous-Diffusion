import matplotlib.pyplot as plt
import numpy as np

# F = np.linspace(0.020,0.050,6) #second phase
# k = np.linspace(0.040,0.070,6)
F = np.linspace(0.01,0.1,6) #first phase
k = np.linspace(0.01,0.1,6)
# # print(F)
# print(k)

fig, ax = plt.subplots(figsize=(len(F), len(k))) #first phase
#fig, ax = plt.subplots(figsize=(8, 8)) #second phase
# Compute spacing between values
df = F[1] - F[0]
dk = k[1] - k[0]
extent1 = min(df, dk) / 2 * 0.98  # slightly less than half to avoid overlap



for i, f_val in enumerate(F):
    for j, k_val in enumerate(k):
        imp_path = f"LF_F{f_val:.3f}_k{k_val:.3f}.png" #so basically it created a new varaible that updates every time we look at other F/k values that we have used.
        try:
            img = plt.imread(imp_path)
            ax.imshow(img, extent=([f_val-extent1, f_val+extent1 , k_val-extent1, k_val + extent1]), aspect='equal')
        #instead of 0.05, we can use 0.004
        except FileNotFoundError:
            pass

gap = 0.0001 #- first phase
# gap = 0.001 #second phase
#orginally it was 0.01

'''ZOOMED'''
# ax.set_xlim(0.020 - 2.5 * gap, F[-1] + gap)#Instead of 2.5 I had used 5, so just wanted to check if it is better sized with 2.5
# ax.set_ylim(0.040 - 2.5 * gap, k[-1] + gap) # This should give me a little space in the graph, 

'''Intial'''
ax.set_xlim(0, F[-1] + gap)#Instead of 2.5 I had used 5, so just wanted to check if it is better sized with 2.5
ax.set_ylim(0, k[-1] + gap) # This should give me a little space in the graph, 
#for the above two lines - when doing the first phase, set those values to 0. For the second phase, we set them to where I want them to start
    #Basically I have just said, the limit of the x and y axis is the last value of F and k plus a little gap.


'''Labeling'''
ax.set_xlabel('F (U Generation Rate)')
ax.set_ylabel('k (V Kill Rate)')
ax.set_xticks(F)
ax.set_yticks(k)
plt.title('Phase Chart of Gray-Scott Model')
# plt.title('Phase Diagram of Gray-Scott Model Zoomed In')

plt.show()


