import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
def mymin(A, B):
    return np.where(A<B,A,B)
    
def bdx(P,m):
    return P-P[[0]+list(range(m-1)),:]

def bdy(P,n):
    return P-P[:,[0]+list(range(n-1))]    

def TV(In,N,dt,var_n):
    ep=0.0001
    J=In.copy()
    m, n = In.shape
    for i in range(N): 
        DfJx=J[list(range(1,m))+[m-1],:]-J
        DbJx=J-J[[0]+list(range(m-1)),:]
        DfJy=J[:,list(range(1,n))+[n-1]]-J
        DbJy=J-J[:,[0]+list(range(n-1))]

        TempDJx=(ep+DfJx*DfJx+((np.sign(DfJy)+np.sign(DbJy))
          *mymin(np.abs(DfJy),np.abs(DbJy))/2)**2)**(1/2)
        TempDJy=(ep+DfJy*DfJy+((np.sign(DfJx)+np.sign(DbJx))
          *mymin(np.abs(DfJx),np.abs(DbJx))/2)**2)**(1/2)
 
        DivJx=DfJx/TempDJx;
        DivJy=DfJy/TempDJy;
 
        Div=bdx(DivJx,m)+bdy(DivJy,n)
        
        lam = max(np.mean(Div*(J-In))/var_n,0)    
        J += dt * Div -dt*lam*(J-In)
    return J

def TV4(In,N,dt,lam):
    ep=0.0001
    J=In.copy()
    m, n = In.shape
    for i in range(N): 
        DfJx=J[list(range(1,m))+[m-1],:]-J
        DfJy=J[:,list(range(1,n))+[n-1]]-J

        TempDJ=(ep+DfJx*DfJx+DfJy*DfJy)**(1/2)
 
        DivJx=DfJx/TempDJ;
        DivJy=DfJy/TempDJ;
 
        Div=bdx(DivJx,m)+bdy(DivJy,n)
          
        J += dt * Div -dt*lam*(J-In)
    return J    

## 基于Joccobi迭代的方法
#def TVer(In, N, lam):
#    # 得到图像大小
#    ep = 0.0001
#    U = In.copy()
#    m, n = U.shape
#    for i in range(N):
#        nto = list(range(1, m))+[m-1]
#        sto = [0]+list(range(m-1))
#        eto = list(range(1, n))+[n-1]
#        wto = [0]+list(range(n-1))
#        Un = U[nto, :]
#        Us = U[sto, :]
#        Ue = U[:, eto]
#        Uw = U[:, wto]
#        Une = U[nto, :][:, eto]
#        Unw = U[nto, :][:, wto]
#        Use = U[sto, :][:, eto]
#        Usw = U[sto, :][:, wto]
#        TUe = 1/np.sqrt((ep + (Ue - U)**2 + (Une + Un - Us - Use)**2 / 16))
#        TUw = 1/np.sqrt((ep + (Uw - U)**2 + (Unw + Un - Us - Usw)**2 / 16))
#        TUn = 1/np.sqrt((ep + (Un - U)**2 + (Unw + Uw - Ue - Une)**2 / 16))
#        TUs = 1/np.sqrt((ep + (Us - U)**2 + (Usw + Uw - Ue - Use)**2 / 16))
#        TU = TUe+TUs+TUw+TUn
#        hoe = TUe/(TU+lam)
#        how = TUw/(TU+lam)
#        hon = TUn/(TU+lam)
#        hos = TUs/(TU+lam)
#        hoo = lam/(TU+lam)
#        U = hoe*Ue + how*Uw + hon*Un + hos*Us + hoo*In
#    return U
#
#def TV3(In, N, lam, dt):
#    # 得到图像大小
#    ep = 0.01
#    U = In.copy()
#    m, n = U.shape
#    for i in range(N):
#        nto = list(range(1, m))+[m-1]
#        sto = [0]+list(range(m-1))
#        eto = list(range(1, n))+[n-1]
#        wto = [0]+list(range(n-1))
#        Un = U[nto, :]
#        Us = U[sto, :]
#        Ue = U[:, eto]
#        Uw = U[:, wto]
#        Du2 = ((Un-Us)**2+(Ue-Uw)**2)/4
#        theta = np.where(Du2>ep**2, 1,ep)
#        U = ((Ue + Uw + Un + Us - U)/theta - lam*(U-In)) *dt + U  
#    return U

if __name__ == '__main__':
    nx, ny = 200, 200
    # Generate image
    img = 100.0*np.ones((nx,ny))
    img[75:150,75:150] = 150.0
    # Adding Gaussian noise
    nmean, nsigma = 0.0, 12.0
    nimg = np.random.normal(nmean,nsigma,(nx,ny)) + img
    #img1 = TVer(nimg, 100, 0.001)
    img2 = TV(nimg, 1000, 0.05, nsigma**2)
    img3 = TV4(nimg, 1000, 0.05, 0.01)
    plt.figure()
    plt.subplot(1,3,1)
    plt.title('Ground Truth')
    plt.imshow(img,"gray")
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title('Noisy Image')
    plt.imshow(nimg,"gray")
    plt.axis('off')
    #plt.subplot(1,4,3)
#    plt.figure()
#    plt.title(r'Denoisies Image 1- $|\nabla\cdot|_1$')
#    plt.imshow(img1,"gray")
#    plt.axis('off')
   # plt.subplot(1,4,4)
#    plt.figure()
#    plt.title(r'Denoisies Image 2- $|\nabla\cdot|_1$')
#    plt.imshow(img2,"gray")
#    plt.axis('off')
#    plt.figure()
    plt.subplot(1,3,3)
    plt.title(r'Denoisies Image 3- $|\nabla\cdot|_1$')
    plt.imshow(img3,"gray")
    plt.axis('off')
    plt.show()