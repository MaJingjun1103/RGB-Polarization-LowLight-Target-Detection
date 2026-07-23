import numpy as np

def rgb_to_lab(img):

    img = img.astype(np.float64) / 255.0

    M = np.array([[0.4123,0.3577,0.1802],
                  [0.2126,0.7153,0.0721],
                  [0.0193,0.1192,0.9493]])

    xyz = img.reshape(-1,3) @ M.T
    X,Y,Z = xyz[:,0], xyz[:,1], xyz[:,2]

    Xn,Yn,Zn = 0.9505,1.0,1.0888

    def f(t):
        d = (6/29)**3
        return np.where(t>d,np.cbrt(t),(1/3)*(29/6)**2*t+4/29)

    fx,fy,fz = f(X/Xn),f(Y/Yn),f(Z/Zn)

    L = 116*fy
    a = 500*(fx-fy)
    b = 200*(fy-fz)

    return np.stack([L,a,b],axis=1).reshape(img.shape)


def pol_to_lab(dolp):

    if len(dolp.shape)==2:
        d = dolp.astype(np.float32)
        d /= (d.max()+1e-8)
        rgb = np.stack([d,d,d],axis=2)
    else:
        rgb = dolp.astype(np.float32)/255.0

    return rgb_to_lab((rgb*255).astype("uint8"))