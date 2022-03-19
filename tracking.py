import cv2
import numpy as np
import glob
import os
import time
import detection

class Tracking:
    def __init__(self):
        self.DC = detection.Detector()
        self.IOU_FLAG = True
        self.idx=0
        self.checkList = []
        self.FIRSTDETECT = True
        self.CONTROL = False
        self.frame = 0
        self.id = 1
        self.predList = []         #[["name",(10,20,30,40), 1]]
        self.tempPredList = self.predList
        self.STUPID_FLAG = False

    def gui(self):
        for fname in sorted(glob.glob("enter_image_directory/*.jpg"), key=os.path.getmtime):       
            s = time.time()
            self.image_rgb  = cv2.imread(fname)
            self.image_rgb=cv2.resize(self.image_rgb,(812,812),interpolation=cv2.INTER_AREA)
            #self.height2, self.width2, d = self.image_rgb.shape

            self.detections, self.detectionImg,_ = self.DC.image_detection(self.image_rgb) 
            self.height, self.width, d = self.image_rgb.shape
            if self.detections != []:
                
                self.run_gui()
                self.FIRSTDETECT = False
            #     print("predlist:", self.predList)
            #     print("---------------------------------")

            cv2.imshow("detection",self.image_rgb)
            #cv2.imshow("tracking",)
            self.frame+=1
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
            print(time.time()-s)
    def run_gui(self):
        for index,pred in enumerate(self.detections):
            self.xmin,self.ymin,self.xmax,self.ymax = self.bbox(pred)
            name = pred[0]
            bbox = (self.xmin,self.ymin,self.xmax,self.ymax)

            if self.FIRSTDETECT == True:
                if self.CONTROL == False:
                    self.predList = [[name, bbox, self.id]]
                    self.CONTROL = True
                    self.id+=1
                else:
                    self.predList = np.vstack((self.predList, [[name, bbox, self.id]]))
                    self.id+=1
            else:
                current_bbox = [[self.xmin, self.ymin], [self.xmax, self.ymin], [self.xmax, self.ymax], [self.xmin, self.ymax]]
                for self.idx, pred2 in enumerate(self.predList):
                    self.txmin = self.predList[self.idx][1][0]
                    self.tymin = self.predList[self.idx][1][1]
                    self.txmax = self.predList[self.idx][1][2]
                    self.tymax = self.predList[self.idx][1][3]

                    temp_bbox = [[self.txmin, self.tymin], [self.txmax, self.tymin], [self.txmax, self.tymax], [self.txmin, self.tymax]]
                    self.iou_score = self.calculate_iou(current_bbox, temp_bbox)
                    self.IOU_FLAG = False
                    if self.iou_score>=0.1:
                        self.predList[self.idx][1] = (self.xmin,self.ymin,self.xmax,self.ymax)
                        self.IOU_FLAG = True 
                        self.checkList = np.append(self.checkList, pred2[2])
                        #cv2.putText(self.image_rgb, str(pred2[2]), (self.xmin,self.ymin-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
                        #cv2.rectangle(self.image_rgb, (self.txmin, self.tymin), (self.txmax, self.tymax), (255,0,0), 2)
                        break

                if self.IOU_FLAG == False:
                    self.predList = np.vstack((self.predList, [[name, bbox, self.id]]))
                    #self.checkList = np.append(self.checkList, len(self.predList)-1)
                    self.id+=1

        for idx, pred3 in enumerate(self.predList):
            #self.xxmin,self.yymin,self.xxmax,self.yymax = self.bbox2(pred3)
            cv2.putText(self.image_rgb, str(pred3[2]), (pred3[1][0],pred3[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
            cv2.rectangle(self.image_rgb, (pred3[1][0], pred3[1][1]), (pred3[1][2], pred3[1][3]), (255,0,0), 2)
            if np.sum(self.checkList==pred3[2])==0 and self.checkList != []:
                self.tempPredList = np.delete(self.predList, idx, 0)
                self.STUPID_FLAG = True
                # print(self.checkList)
        if self.FIRSTDETECT == False and self.STUPID_FLAG == True:
            self.predList = self.tempPredList
        self.checkList = []

    def bbox(self, pred):
        bbox_width = (pred[2][2]*self.width)/416
        bbox_height = (pred[2][3]*self.height)/416
        center_x = (pred[2][0]*self.width)/416
        center_y = (pred[2][1]*self.height)/416

        xmin = int(center_x - (bbox_width / 2))
        ymin = int(center_y - (bbox_height / 2))
        xmax = int(center_x + (bbox_width / 2))
        ymax = int(center_y + (bbox_height / 2))
        return xmin, ymin, xmax, ymax

    def bbox2(self, pred):
        bbox_width = (pred[1][2]*self.width)/416
        bbox_height = (pred[1][3]*self.height)/416
        center_x = (pred[1][0]*self.width)/416
        center_y = (pred[1][1]*self.height)/416

        xmin = int(center_x - (bbox_width / 2))
        ymin = int(center_y - (bbox_height / 2))
        xmax = int(center_x + (bbox_width / 2))
        ymax = int(center_y + (bbox_height / 2))
        return xmin, ymin, xmax, ymax

    def calculate_iou(self, first_bb_points, second_bb_points):
        stencil = np.zeros(self.image_rgb.shape).astype(self.image_rgb.dtype)
        contours = [np.array(first_bb_points)]
        color = [255, 255, 255]
        cv2.fillPoly(stencil, contours, color)
        result1 = cv2.bitwise_and(self.image_rgb, stencil)
        result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2RGB)

        stencil = np.zeros(self.image_rgb.shape).astype(self.image_rgb.dtype)
        contours = [np.array(second_bb_points)]
        color = [255, 255, 255]
        cv2.fillPoly(stencil, contours, color)
        result2 = cv2.bitwise_and(self.image_rgb, stencil)
        result2 = cv2.cvtColor(result2, cv2.COLOR_BGR2RGB)

        intersection = np.logical_and(result1, result2)
        union = np.logical_or(result1, result2)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

if __name__ == "__main__":
    track = Tracking()
    track.gui()