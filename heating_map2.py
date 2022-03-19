import cv2
import numpy as np
import time
import glob
import os
from detection import Detector

class Heating:
    def __init__(self) -> None:
        self.DC = Detector()
        self.flag=True
        self.heatMapImg = np.zeros((960,1280),dtype='uint8')
        self.length=37#23
        self.sig=7 #3
        #self.kernel = self.gaussian_kernel()
        self.frame = 0

    def gui(self):
        cap = cv2.VideoCapture("data/PTZ_1/Derince/08.20 - 08.30.avi")
        while (cap.isOpened()):
            #time.sleep(0.04)
            ret, image_org  = cap.read()
            image_org=cv2.resize(image_org,(1280,960),interpolation=cv2.INTER_AREA)
            self.h, self.w,_ = image_org.shape
            
            self.detection = self.DC.main_deepsort(image_org)
            if self.detection != []:
                midPts=self.makePoints()
            if self.frame%2==0:
                for ptsx, ptsy, rng in midPts:
                    self.heat_map(ptsx,ptsy, rng) 

                HImg=cv2.applyColorMap(self.heatMapImg, cv2.COLORMAP_JET)
                #HImg = cv2.cvtColor(HImg, cv2.COLOR_BGR2RGB)
                HImg[HImg[:,:,0]==128]=0

            merImg = cv2.addWeighted(image_org, 0.7, HImg, 0.3, 0)  #2,-1
            cv2.imshow('trr',cv2.resize(HImg,(int(self.w),int(self.h))))
            cv2.imshow('mer',cv2.resize(merImg,(int(self.w),int(self.h))))
            cv2.waitKey(10)
            self.frame+=1
            self.heatMapImg = np.zeros((960,1280),dtype='uint8')
            
    def makePoints(self):
        midPoints = []
        midPoints = [self.calc_bbox(pred) for i, pred in enumerate(self.detection)]
        return midPoints


    def calc_bbox(self, pred):
        bbox_width = (pred[2][2]*self.w)/416
        bbox_height = (pred[2][3]*self.h)/416
        center_x = (pred[2][0]*self.w)/416
        center_y = (pred[2][1]*self.h)/416
        self.xmin = int(center_x - (bbox_width / 2))
        self.ymin = int(center_y - (bbox_height / 2))
        self.xmax = int(center_x + (bbox_width / 2))
        self.ymax = int(center_y + (bbox_height / 2))

        self.rangee = (self.xmax-self.xmin)*(self.ymax-self.ymin)

        xmid, ymid = ((self.xmax-self.xmin)+self.xmin, self.ymax)
        return  self.xmin, self.ymin, self.rangee # xmid-20, ymid, self.rangee

    def heat_map(self, xM, yM, rng):
        self.kernel = self.gaussian_kernel(rng)
        if xM>660:
            y_offset= yM-20
            x_offset= xM+10
        else:
            y_offset= yM
            x_offset= xM-50

        y1, y2 = y_offset, y_offset + self.kernel.shape[0]
        x1, x2 = x_offset, x_offset + self.kernel.shape[1]

            


        # x1 = xM
        # y1 = yM

        # x1 =np.clip(x1, 0, 416)
        # x2 =np.clip(x2, 0, 416)
        # y1 =np.clip(y1, 0, 416)
        # y2 =np.clip(y2, 0, 416)
        try:
            self.heatMapImg[y1:y2, x1:x2] = np.clip(self.heatMapImg[y1:y2, x1:x2]+(self.kernel[:, :]),0,255)
        except:
            pass
            #print("error")
    def gaussian_kernel(self, rng):
        if rng<=200:
            self.length=167#23   #13
            self.sig=33 #3       #3
        else:
            self.length=167#23        #23
            self.sig=33 #3            #3
        ax = np.linspace(-(self.length - 1) / 2., (self.length - 1) / 2., self.length)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(self.sig))
        kernel = np.outer(gauss, gauss)*20  #3
        return kernel

if __name__ == "__main__":
    H = Heating()
    H.gui()


    # org = np.zeros((416,416),dtype='uint8')
    # l=37
    # sig=7
    # ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    # gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    # kernel = np.outer(gauss, gauss)*255
    # # kernel=cv2.applyColorMap(kernel.astype('uint8'), cv2.COLORMAP_JET)
    # # cv2.imshow('trr',kernel.astype('uint8'))
    # # cv2.waitKey(0)

    # y_offset=77
    # x_offset=100
    # y1, y2 = y_offset, y_offset + kernel.shape[0]
    # x1, x2 = x_offset, x_offset + kernel.shape[1]
    # org[y1:y2, x1:x2] = np.clip(org[y1:y2, x1:x2]+(kernel[:, :]),0,255)

    # # y_offset=130
    # # x_offset=110
    # # y1, y2 = y_offset, y_offset + kernel.shape[0]
    # # x1, x2 = x_offset, x_offset + kernel.shape[1]
    # # org[y1:y2, x1:x2] = np.clip(org[y1:y2, x1:x2]+(kernel[:, :]),0,255)

    # org = np.clip(org, 0, 255)
    # org=cv2.applyColorMap(org, cv2.COLORMAP_JET)
    # cv2.imshow('trr',org)
    # cv2.waitKey(0)

