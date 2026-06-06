import matplotlib.pyplot as plt
import numpy as np
import os


F = np.linspace(0.020,0.050,6) #second phase
k = np.linspace(0.040,0.070,6)
# F = np.linspace(0.01,0.1,6) #first phase
# k = np.linspace(0.01,0.1,6)
print(F)
print(k)

image_folder = "Zoomed_Phase_copy"
fig, ax = plt.subplots(figsize=(len(F), len(k))) #first phase
#fig, ax = plt.subplots(figsize=(8, 8)) #second phase
extent1 = 0.003 #Use 0.00325 if it does not work

for i, f_val in enumerate(F):
    for j, k_val in enumerate(k):
        imp_path = os.path.join(image_folder,f"LF_F{f_val:.3f}_k{k_val:.3f}.png") #so basically it created a new varaible that updates every time we look at other F/k values that we have used. Also trying to incorpate my folders
        try:
            img = plt.imread(imp_path)
            ax.imshow(img, extent=([f_val-extent1, f_val+extent1 , k_val-extent1, k_val + extent1]), aspect='auto')
        #instead of 0.05, we can use 0.004
        except FileNotFoundError:
            pass

#gap = 0.001 - first phase
gap = 0.001 #second phase
constant = 5

#orginally it was 0.01
ax.set_xlim(0.020 - constant * gap, F[-1] + gap)#Instead of 2.5 I had used 5, so just wanted to check if it is better sized with 2.5
ax.set_ylim(0.040 - constant * gap, k[-1] + gap) # This should give me a little space in the graph, 
#for the above two lines - when doing the first phase, set those values to 0. For the second phase, we set them to where I want them to start
    #Basically I have just said, the limit of the x and y axis is the last value of F and k plus a little gap.
ax.set_xlabel('F (U Generation Rate)')
ax.set_ylabel('k (V Kill Rate)')
ax.set_xticks(F)
ax.set_yticks(k)
#plt.title('Phase Chart of Gray-Scott Model')
plt.title('Phase Diagram of Gray-Scott Model Zoomed In')
plt.show()