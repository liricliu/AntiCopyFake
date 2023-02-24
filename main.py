import matplotlib.pylab as plt
import matplotlib.image as mpimg
import numpy as np
import scipy as sci
from PIL import Image
import copy
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
        # return Image.fromarray(new_arr)
        return new_arr

def main():

    #读入图片
    img = Image.open("/home/liricmechan/antifake/test.jpg").convert("RGB")
    im = np.array(img)
    width=im.shape[1]
    height=im.shape[0]
    channel=im.shape[2]
    print("读入的图像信息：\n宽",width,"高",height,"色彩通道数",channel)

    # 生成高斯金字塔
    GaussianPyra=[]
    for i in range(0,4):
        GaussianPyra.append(GaussianBlur(sigma=pow(2,i)).filter(img))
    # 生成高斯差分金字塔
    DoGPyra=[]
    for i in range(0,3):
        DoGPyra.append(GaussianPyra[i+1]-GaussianPyra[i])
    # 输出高斯金字塔图像
    for index,oct in enumerate(GaussianPyra):
        Image.fromarray(oct).save("PicOutput/testGauss"+str(index)+".jpg");
    # 输出高斯差分金字塔图像
    for index,oct in enumerate(DoGPyra):
        Image.fromarray(oct).save("PicOutput/testDoG"+str(index)+".jpg");

    # 查找极值点
    PoleSpotR=[]
    PoleSpotG=[]
    PoleSpotB=[]
    is_max_r=True
    is_min_r=True
    is_max_g=True
    is_min_g=True
    is_max_b=True
    is_min_b=True
    im_r=copy.copy(im)
    im_g=copy.copy(im)
    im_b=copy.copy(im)
    for k in range(0,3):
        for i in range(1,height-1):
            for j in range(1,width-1):
                is_max_r=True
                is_min_r=True
                is_max_g=True
                is_min_g=True
                is_max_b=True
                is_min_b=True
                for i_k in range (-1,2):
                    for j_k in range (-1,2):
                        for k_k in range(0 if k==0 else -1,1 if k==2 else 2):
                            if(not (i_k==0 and j_k==0 and k_k==0)):
                                if (DoGPyra[k])[i,j,0]>=(DoGPyra[k+k_k])[i+i_k,j+j_k,0]:
                                    is_min_r=False
                                if (DoGPyra[k])[i,j,0]<=(DoGPyra[k+k_k])[i+i_k,j+j_k,0]:
                                    is_max_r=False
                                if (DoGPyra[k])[i,j,0]>=(DoGPyra[k+k_k])[i+i_k,j+j_k,1]:
                                    is_min_g=False
                                if (DoGPyra[k])[i,j,0]<=(DoGPyra[k+k_k])[i+i_k,j+j_k,1]:
                                    is_max_g=False
                                if (DoGPyra[k])[i,j,0]>=(DoGPyra[k+k_k])[i+i_k,j+j_k,2]:
                                    is_min_b=False
                                if (DoGPyra[k])[i,j,0]<=(DoGPyra[k+k_k])[i+i_k,j+j_k,2]:
                                    is_max_b=False
                if(is_max_r==True or is_min_r==True):
                    PoleSpotR.append([i,j,k])
                    im_r[i,j,0]=255
                    im_r[i,j,1]=0
                    im_r[i,j,2]=0
                if(is_max_g==True or is_min_g==True):
                    PoleSpotG.append([i,j,k])
                    im_g[i,j,0]=0
                    im_g[i,j,1]=255
                    im_g[i,j,2]=0
                if(is_max_b==True or is_min_b==True):
                    PoleSpotB.append([i,j,k])
                    im_b[i,j,0]=0
                    im_b[i,j,1]=0
                    im_b[i,j,2]=255
    # 输出R分量极值点图
    Image.fromarray(im_r).save("PicOutput/RChannel"+".jpg");
    # 输出G分量极值点图
    Image.fromarray(im_g).save("PicOutput/GChannel"+".jpg");
    # 输出B分量极值点图
    Image.fromarray(im_b).save("PicOutput/BChannel"+".jpg");

    print(PoleSpotR)
    print(PoleSpotG)
    print(PoleSpotB)
    #print(DoGPyra)

    #plt.imshow(Image.fromarray(img))
    #plt.axis('off')
    #plt.show()

main()