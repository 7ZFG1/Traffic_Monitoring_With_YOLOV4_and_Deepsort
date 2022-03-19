import cv2
import numpy as np
from detection import Detector
import time
import random

class FixMatch:
    def __init__(self):
        self.STUPID_FLAG = False
        self.file = []
        self.i=0
        self.filename = 0
        self.DC = Detector()
        self.var = 30
        self.deviation = self.var*0.5
        self.a = 0.3
        self.b = 1.7
        self.label_list = {"car": 0,
            "truck": 1,
            "van": 2,
            "bus": 3,
            "motorbike": 4,
            "pedestrian":5
            }

    def run_gui(self):
        strongIMG = self.strongAug(self.image_rgb)
        weakIMG = self.strongAug(self.image_rgb)

        self.detections_strong, self.OBJDetection_strong = self.DC.image_detection(strongIMG)
        self.detections_weak, self.OBJDetection_weak = self.DC.image_detection(weakIMG)
        
        for predStrong, predWeak in zip(self.detections_strong, self.detections_weak):
            if predStrong[0] == predWeak[0] and predStrong[0] != 'big-truck' and predWeak[0] != 'big-truck':
                self.STUPID_FLAG = True
                print("SAME! ", predStrong[0], self.i)
                bbox_width = predStrong[2][2]/416
                bbox_height = predStrong[2][3]/416
                center_x = predStrong[2][0]/416
                center_y = predStrong[2][1]/416 
                self.file = open("enter_image_dir/0000" + str(self.i) + ".txt", 'a')
                caption = "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(self.label_list[predStrong[0]], center_x, center_y, bbox_width, bbox_height)
                self.file.writelines(caption + "\n")

            else:
                pass
                #print("NOT SAME!! ", predStrong[0], predWeak[0])
        if self.STUPID_FLAG:
            self.file.close()
            self.STUPID_FLAG = False

    def strongAug(self, strong_image):
        Red = strong_image[:,:,2]
        Green = strong_image[:,:,1]
        Blue = strong_image[:,:,0]

        strong_aug = np.ones((416,416,3), dtype=np.uint8)

        strong_aug[:,:,0] = Blue*random.uniform(self.a, self.b)
        strong_aug[:,:,1] = Green*random.uniform(self.a, self.b)
        strong_aug[:,:,2] = Red*random.uniform(self.a, self.b) 
        return strong_aug

    def weakAug(self, weak_image):
        noise = np.random.normal(0, self.deviation, weak_image.shape)
        weak_image = weak_image + noise.astype("uint8")
        np.clip(weak_image, 0., 255.)
        return weak_image

    def labeling(self):
        pass

    def gui(self):
        while True:
            time.sleep(0.02)
            start= time.time()
            
            image = cv2.imread("enter_image_dir/0000" + str(self.i) +".jpg")

            self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            self.image_rgb = image
            self.image_rgb = cv2.resize(self.image_rgb,(416,416),interpolation=cv2.cv2.INTER_AREA)

            
            self.run_gui()
            
            # img = self.strongAug(self.image_rgb)
            # self.detections, self.OBJDetection = self.DC.image_detection(img)

            # cv2.imshow("detection",self.OBJDetection)
            # cv2.waitKey(10)
            #self.filename += 1
            self.i+=1
            fps = int(1 / (time.time() - start))
            #print("FPS: {}".format(fps))

if __name__ == "__main__":
    FM = FixMatch()
    FM.gui()











