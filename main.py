from PIL import Image
from numpy import *
from pylab import *
import scipy as sci
import os
class GaussianBlur(object):
    def __init__(self, kernel_size=3, sigma=1):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self.gaussian_kernel()
 
    def gaussian_kernel(self):
        kernel = np.zeros(shape=(self.kernel_size, self.kernel_size), dtype=float)
        radius = self.kernel_size//2
        for y in range(-radius, radius + 1):  # [-r, r]
            for x in range(-radius, radius + 1):
                # 二维高斯函数
                v = 1.0 / (2 * np.pi * self.sigma ** 2) * np.exp(-1.0 / (2 * self.sigma ** 2) * (x ** 2 + y ** 2))
                kernel[y + radius, x + radius] = v  # 高斯函数的x和y值 vs 高斯核的下标值
        kernel2 = kernel / np.sum(kernel)
        return kernel2
 
    def filter(self, img: Image.Image):
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            new_arr = sci.signal.convolve2d(img_arr, self.kernel, mode="same", boundary="symm")
        else:
            h, w, c = img_arr.shape
            new_arr = np.zeros(shape=(h, w, c), dtype=float)
            for i in range(c):
                new_arr[..., i] = sci.signal.convolve2d(img_arr[..., i], self.kernel, mode="same", boundary="symm")
        new_arr = np.array(new_arr, dtype=np.uint8)
        return Image.fromarray(new_arr)
        # return new_arr

def process_image(imagename,resultname,params="--edge-thresh 10 --peak-thresh 5"):
    """ Process an image and save the results in a file. """

    if imagename[-3:] != 'pgm':
        # create a pgm file
        im = Image.open(imagename).convert('L') 
        im.save('tmp.pgm')                      
        imagename = 'tmp.pgm'
   
    cmmd = str("E:\SIFT-Python-master\sift.exe "+imagename+" --output="+resultname+
                " "+params)
    os.system(cmmd)                              
    print ('processed', imagename, 'to', resultname)


def read_features_from_file(filename):
    """ Read feature properties and return in matrix form. """
    
    f = loadtxt(filename)
    return f[:,:4],f[:,4:] # feature locations, descriptors


def plot_features(im,locs,circle=True):
    """ Show image with features. input: im (image as array), 
        locs (row, col, scale, orientation of each feature). """

    def draw_circle(c,r):
        t = arange(0,1.01,.01)*2*pi
        x = r*cos(t) + c[0]
        y = r*sin(t) + c[1]
        plot(x,y,'b',linewidth=2)

    imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2],p[2]) 
    else:
        plot(locs[:,0],locs[:,1],'ob')
    axis('off')

 

if __name__ == '__main__':

    yuzhi=0.1

    #imgpre = Image.open("E:\\SIFT-Python-master\\test.jpg").convert("RGB")
    #GaussianBlur(sigma=1).filter(imgpre).save("E:\\SIFT-Python-master\\t2.jpg")
    #print("Filtered")
    imname = ('E:\\SIFT-Python-master\\t5.jpg')          
    im=Image.open(imname)
    process_image(imname,'test.sift')

    l1,d1 = read_features_from_file('test.sift')       
    figure()
    gray()
    
    plot_features(im,l1,circle = True)
    xxxxxxx=[-1,-1]
    pipeilist=[xxxxxxx]
    rangex=d1.shape[0]

    for i_i in range(0,rangex):
        i_j_d_min=999999999
        i_j_d_min1=999999999
        j_i_min=-1
        j_i_min1=-1
        for j_i in range(i_i+1,rangex):
            i_j_d=0
            for k in range(0,128):
                i_j_d=i_j_d+(d1[i_i,k]-d1[j_i,k])*(d1[i_i,k]-d1[j_i,k])
            if i_j_d<i_j_d_min:
                i_j_d_min=i_j_d
                j_i_min=j_i
            else:
                if i_j_d<i_j_d_min1:
                    i_j_d_min1=i_j_d
                    j_i_min1=j_i
        if i_j_d_min/i_j_d_min1<yuzhi:
            print(i_j_d_min/i_j_d_min1,i_j_d_min,i_j_d_min1)
            pipeilist.append([i_i,j_i_min])
            print("pipeichenggong:"+str(i_i)+","+str(j_i_min))
            print(l1[i_i,0],l1[i_i,1],l1[j_i_min,0],l1[j_i_min,1],l1[j_i_min1,0],l1[j_i_min1,1])
            plot([l1[i_i,0],l1[j_i_min,0]],[l1[i_i,1],l1[j_i_min,1]])
    #plot([100,150],[500,200])
    #print(pipeilist)
    #plot(locs[:,0],locs[:,1],'ob')


            
    #for i in pipeilist:
    #    plot([l1[i[0],0],l1[i[0],1]],[l1[i[1],0],l1[i[1],1]])
    title('sift-features0.1')
    show()