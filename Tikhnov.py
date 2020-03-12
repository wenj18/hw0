# author :Liu Haibo
import numpy as np
from  PIL import Image
nx, ny = 200, 200 # Generate image
img = 100.0*np.ones((nx,ny))
img[75:150,75:150] = 150.0 # Adding Gaussian noise
nmean,nsigma  = 0.0, 12.0
nimg = np.random.normal(nmean,nsigma,(nx,ny)) + img
alpha=100.0*np.ones((nx,ny))
alpha[0:200,0:200] = 0.1
lamd=0.1
u = nimg.copy()
v=nimg.copy()
for i in range(1,100):
    for x in range(0, 199):
        for y in range(0, 199):
           u[x, y] = v[x, y] - alpha[x, y] * (v[x , y]-nimg[x,y]+ lamd*v[x, y])
    v=u.copy()
_u=Image.fromarray(u)
_u = _u.convert('RGB')
_u.save('恢复图像.jpg')
Image._show(_u)