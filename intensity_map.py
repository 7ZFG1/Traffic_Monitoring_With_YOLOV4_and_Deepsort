import cv2
import numpy as np
import matplotlib.path as mplPath
import glob
import os
import time

from detection import Detector

class Intensity_Road:
    def __init__(self) -> None:
        self.DC = Detector()
        self.frame=0
        self.a1 = [[330,362],[424,219],[440,161],[466,157],[472,233],[418,370]]     ####Coordinates provided by images
        self.a2 = [[3,751],[324,371],[414,375],[86,958]]
        self.b1 = [[479,382],[482,151],[513,153],[548,294],[553,294],[552,388]]
        self.b2 = [[300,965],[478,385],[554,387],[530,950]]
        self.c1 = [[597,384],[558,222],[524,148],[554,148],[645,290],[685,384]]
        self.c2 = [[630,953],[594,388],[684,387],[883,951]]
        self.d1 = [[746,375],[637,220],[569,153],[592,156],[697,242],[816,376]]
        self.d2 = [[1060,955],[749,380],[819,378],[1272,903],[1278,950]]

        self.a1_path = mplPath.Path(np.array(self.a1))
        self.b1_path = mplPath.Path(np.array(self.b1))
        self.c1_path = mplPath.Path(np.array(self.c1))
        self.d1_path = mplPath.Path(np.array(self.d1))
        self.a2_path = mplPath.Path(np.array(self.a2))
        self.b2_path = mplPath.Path(np.array(self.b2))
        self.c2_path = mplPath.Path(np.array(self.c2))
        self.d2_path = mplPath.Path(np.array(self.d2))

        self.a1cnt=0
        self.b1cnt=0
        self.c1cnt=0
        self.d2cnt=0
        self.a2cnt=0
        self.b2cnt=0
        self.c2cnt=0
        self.d2cnt=0

        self.colors =[(0,255,0),(255, 0, 0),(0,0,255)]

    def gui(self):
        for fname in sorted(glob.glob("enter_image_dir/*.jpg"), key=os.path.getmtime): 
            time.sleep(0.01)
            image_rgb  = cv2.imread(fname)
            image_rgb=cv2.resize(image_rgb,(1280,960),interpolation=cv2.INTER_AREA)
            self.h, self.w,_= image_rgb.shape
            detected_img, self.detections, self.blackimg = self.DC.main_intensity(image_rgb)

            if self.frame%30==0:
                intensity_level=self.intensity()

            mask=self.mask_img(detected_img, intensity_level)
            
            mask=cv2.resize(mask,(2592,1520),interpolation=cv2.INTER_AREA)
            mask = cv2.rectangle(mask, (1200,60), (1730, 0), (255,255,255), -1)
            cv2.putText(mask, 'Traffic Density', (1200,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
            cv2.imshow("Inference", mask)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
            self.frame+=1

    def mask_img(self, image, intensity_level):
        h, w, d = image.shape
        mask = np.zeros([h,w],dtype=np.uint8)

        a1 = np.array(self.a1,  np.int32)
        b1 = np.array(self.b1,  np.int32)
        c1 = np.array(self.c1,  np.int32)
        d1 = np.array(self.d1,  np.int32)
        a2 = np.array(self.a2,  np.int32)
        b2 = np.array(self.b2,  np.int32)
        c2 = np.array(self.c2,  np.int32)
        d2 = np.array(self.d2,  np.int32)

        mask = np.dstack((mask, mask, mask))

        cv2.fillPoly(mask, [a1], self.colors[int(intensity_level[0])])
        cv2.fillPoly(mask, [b1], self.colors[int(intensity_level[1])])
        cv2.fillPoly(mask, [c1], self.colors[int(intensity_level[2])])
        cv2.fillPoly(mask, [d1], self.colors[int(intensity_level[3])])
        cv2.fillPoly(mask, [a2], self.colors[int(intensity_level[4])])
        cv2.fillPoly(mask, [b2], self.colors[int(intensity_level[5])])
        cv2.fillPoly(mask, [c2], self.colors[int(intensity_level[6])])
        cv2.fillPoly(mask, [d2], self.colors[int(intensity_level[7])])

        mask2 = cv2.bitwise_or(image, mask)
        # mask = cv2.bitwise_or(self.blackimg, mask)
        # mask = self.transform(mask)
        # #mask[0:100,0:1280]=0
        # cv2.imshow("maske", mask)
        # cv2.waitKey(10)
        return mask2
    
    def count_vehicle(self):
        intensity_list=[]
        self.a1cnt=0
        self.b1cnt=0
        self.c1cnt=0
        self.d1cnt=0
        self.a2cnt=0
        self.b2cnt=0
        self.c2cnt=0
        self.d2cnt=0
        for i,pred in enumerate(self.detections):
            x,y = self.bbox(pred)
            point=(int(x),int(y))

            if self.a1_path.contains_point(point):
                self.a1cnt+=1
            elif self.b1_path.contains_point(point):
                self.b1cnt+=1
            elif self.c1_path.contains_point(point):
                self.c1cnt+=1
            elif self.d1_path.contains_point(point):
                self.d1cnt+=1
            elif self.a2_path.contains_point(point):
                self.a2cnt+=1
            elif self.b2_path.contains_point(point):
                self.b2cnt+=1
            elif self.c2_path.contains_point(point):
                self.c2cnt+=1
            elif self.d2_path.contains_point(point):
                self.d2cnt+=1
            else:
                pass
                #print("No vehicle detected on the roads")

        intensity_list= [self.a1cnt, self.b1cnt, self.c1cnt, self.d1cnt, self.a2cnt, self.b2cnt, self.c2cnt, self.d2cnt]
        return intensity_list

    def intensity(self):
        intensity_level=[]
        intensity_list = self.count_vehicle()
        #print(intensity_list)
        for i,ints in enumerate(intensity_list):
            if ints<=2:
                intensity_level = np.append(intensity_level, 0)
            elif 2<ints<=8:
                intensity_level = np.append(intensity_level, 1)
            else:
                intensity_level = np.append(intensity_level, 2)
        #print(intensity_level)
        return intensity_level

    def bbox(self, pred):
        center_x = (pred[2][0]*self.w)/416
        center_y = (pred[2][1]*self.h)/416
        return center_x, center_y

    def transform(self, img):
        #input_pts = np.float32([[1134 , 860],[158 , 874],[744 , 489],[508 , 489]])
        input_pts = np.float32([[1280,960],[0,960],[0 , 0],[0 , 0]])
        output_pts = np.float32([[1280,960],[0,960],[617 , 160],[436 , 160]])
        M = cv2.getPerspectiveTransform(input_pts,output_pts)
        persv_img = cv2.warpPerspective(img,M,(img.shape[1], img.shape[0]),flags=cv2.INTER_LINEAR)
        return persv_img

if __name__ == "__main__":
    intensity = Intensity_Road()
    intensity.gui()