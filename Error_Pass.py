from fonctions.Tensor_Tucker import *
from PIL import Image
from tensorly.decomposition import tucker
import tensorly as tl 
import numpy as np 

image = Image.open('1200x768_le-panda-roux-est-.webp')
X = np.asarray(image)



L = [[100, 300, 3],[300, 500, 3], [600, 800, 3],[800, 900, 3]]
err2= []
for elt in L:
    X_hat_2pass,_,_ = Two_Pass(X, elt)
    err2.append(regret(X, X_hat_2pass, elt))


#error on One Pass
T = [[50, 20, 3], [100, 80, 3], [300, 200, 3], [600, 300, 3]]
err1 = []
for elt in T:
    X_hat_1pass,_ = One_Pass(X, elt,2*np.array(elt)+1 )
    err1.append(regret(X, X_hat_1pass, elt))



fig, ax = plt.subplots(1, 1)
ax.scatter(np.arange(len(err1)), err1, color='#11accd', s=30,
               label=r'$OnePass$', marker='v')
ax.scatter(np.arange(len(err1)), err2, color='#807504', s=30, label=r'$TwoPass$',
               marker='o')
ax.plot(np.arange(len(err1)), err1, color='#11accd', linewidth=1)
ax.plot(np.arange(len(err1)), err2, color='#807504', linewidth=1)


ax.set_title('Difference between One Pass and Two Pass')

plt.legend()
plt.tight_layout()
plt.show()