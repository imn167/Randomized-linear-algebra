import fonctions.rsvd as rd
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.misc import face 
from PIL import Image 
import tensorly as tl 


def hosvd_bis(X,R, e):
    N = len(X.shape)
    fibers = []
    core = X
    for mode in range(N):
        A,_,_ = rd.rsvd(tl.base.unfold(X, mode), R[mode], e=e) #random_state = None
        fibers.append(A)
        core = tl.tenalg.mode_dot(core, A.T, mode)
    return core, fibers

from PIL import Image
image = Image.open('image.jpg')
X = np.asarray(image)


I = np.array([20,50,100,200])
E = []

fig, axarr = plt.subplots(1, 4)
fig.set_size_inches(15, 4)

for l in range(len(I)):
    G, fibers = hosvd_bis(X, [I[l],I[l],3], 0.0005)

    X_hat = tl.tucker_to_tensor((G,fibers))
    
    Err = X-X_hat
    m,n,p = Err.shape
    sum = 0
    for i in range(m):
        for j in range(n):
            for k in range(p):
                sum = sum + Err[i][j][k]**2
    err = np.sqrt(sum) / tl.norm(X, order = 2)
    E.append(err)

    X_hat = X_hat.astype('int8')
    image_stream = Image.fromarray(X_hat, 'RGB')
    axarr[l].imshow(image_stream)
    axarr[l].set_title(I[l])
plt.show()

print(E)
plt.plot(I,E)
plt.scatter(I,E,marker = '*')
plt.title('Relative error $\|X-\hat{X} \| / |X\|$')
plt.show()