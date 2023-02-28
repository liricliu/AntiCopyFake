from PIL import Image
from numpy import *
from pylab import *
import scipy as sci
import os
import pysift
import cv2
from sklearn.decomposition import PCA

yuzhi=0.13 #匹配特征值用的阈值
size_shangxian=3.5 #筛除size大于上限的特征点
wupipei_canshu=100 #降低误匹配率用的

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
            if p[2]<=size_shangxian:
                draw_circle(p[:2],p[2]) 
    else:
        plot(locs[:,0],locs[:,1],'ob')
    axis('off')

if __name__ == '__main__':

    imname = ('测试结果/t4.jpg')          
    im=Image.open(imname)
    image = cv2.imread(imname, 0)
    print("Scanning for feature points")
    l1=[]
    d1=[]
    keypoints, des = pysift.computeKeypointsAndDescriptors(image)
    keypoint_counter=0
    selected_keypoint_counter=0
    for k in keypoints:
        l1.append([k.pt[0],k.pt[1],k.size])
        d1.append(des[keypoint_counter])
        keypoint_counter+=1
        # Filter the large points
        if k.size<size_shangxian:
            selected_keypoint_counter+=1
    l1=np.array(l1)
    d1=np.array(d1)
    pca = PCA(n_components=20)
    pca.fit(d1)
    d1=pca.fit_transform(d1)
    print(str(keypoint_counter)+' feature points detected.')
    print(str(selected_keypoint_counter)+' feature points selected.')

    fig=figure()
    gray()
    
    plot_features(im,l1,circle = True)
    rangex=d1.shape[0]
    counter=0

    for i_i in range(0,rangex):
        i_j_d_min=999999999
        i_j_d_min1=999999999
        j_i_min=-1
        j_i_min1=-1
        if l1[i_i,2]>3.5:
            continue
        for j_i in range(i_i+1,rangex):
            i_j_d=0
            for k in range(0,20):
                i_j_d=i_j_d+(d1[i_i,k]-d1[j_i,k])*(d1[i_i,k]-d1[j_i,k])
            if i_j_d<i_j_d_min:
                i_j_d_min=i_j_d
                j_i_min=j_i
            else:
                if i_j_d<i_j_d_min1:
                    i_j_d_min1=i_j_d
                    j_i_min1=j_i
        if i_j_d_min/i_j_d_min1<yuzhi and abs(i_i-j_i_min)>keypoint_counter/wupipei_canshu:
            print("Matched Index:("+str(i_i)+","+str(j_i_min)+"),min/min1="+str(i_j_d_min/i_j_d_min1))
            plot([l1[i_i,0],l1[j_i_min,0]],[l1[i_i,1],l1[j_i_min,1]])
            counter+=1
    print(str(counter)+" points matched in total.")
    title('Output')
    fig.savefig('Output.jpg')
    show()