import cv2
import time
import os
import glob
import numpy as np
from detection import Detector
from my_deepsort import Tracking
import matplotlib.path as mplPath
import pandas as pd

class Vehicle_Count:
    def __init__(self): 
        self.indexes = ['1','2','3','4','5','6','7','8']
        self.main_matrix = np.zeros((8,8))
        self.flag1 = False
        self.flag2 = False
        self.tracking = Tracking()       
        self.area1 = [[721,351],[823,345],[902,391],[779,418]]         ###Coordinates provided by images
        self.area2 = [[98,583],[222,518],[337,533],[204,674]]

        self.area3 = [[657,927],[1384,759],[1488,851],[741,993]]
        self.area4 = [[1370,719],[1593,642],[1744,727],[1504,823]]

        #self.area5 = [[1004,319],[1164,330],[1223,431],[972,419]]

        self.area6 = [[858,362], [949,345],[1069,392],[952,422]]
        self.area7 = [[577,431], [560,409],[684,384],[727,402]]
        self.area8 = [[571,455], [558,485],[660,495],[682,453]]

        self.area1_path = mplPath.Path(np.array(self.area1))
        self.area2_path = mplPath.Path(np.array(self.area2))
        self.area3_path = mplPath.Path(np.array(self.area3))
        self.area4_path = mplPath.Path(np.array(self.area4))
        #self.area5_path = mplPath.Path(np.array(self.area5))
        self.area6_path = mplPath.Path(np.array(self.area6))
        self.area7_path = mplPath.Path(np.array(self.area7))
        self.area8_path = mplPath.Path(np.array(self.area8))

        self.id_list1 = []
        self.id_list2 = []
        self.id_list3 = []
        self.id_list4 = []
        self.id_list5 = []
        self.id_list6 = []
        self.id_list7 = []
        self.id_list8 = []

        self.in_out = []

    def gui(self, detections, image):
        self.h, self.w, d = image.shape
        img = self.run_gui2(detections, image)
        img = self.mask_img(img)
        
        return img

    def run_gui2(self, detections, image):  
        tr_bbox, ids, pred = self.tracking.run_gui4out(image, detections)  
        if tr_bbox!=[]: 
            for i, bx in enumerate(tr_bbox[:len(pred)]):
                xmin=bx[0]
                ymin=bx[1]
                xmax=bx[2]
                ymax=bx[3]
                xmid = int((xmax+xmin)/2)
                ymid = int((ymax+ymin)/2)
                cv2.putText(image_rgb, str(ids[i]), (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                cv2.circle(image, (xmid,ymid),2,(0,0,0),-1)
                self.area_check(xmid,ymid,ids[i])
                
        #print(self.in_out)
        print(self.main_matrix)
        df = pd.DataFrame (self.main_matrix, columns=self.indexes)
        df.index=self.indexes
        filepath = 'output.xlsx'

        df.to_excel(filepath, index=True)
        return image

    def mask_img(self, image):
        h, w, d = image.shape
        mask = np.zeros([h,w],dtype=np.uint8)

        area1 = np.array(self.area1,  np.int32)
        area2 = np.array(self.area2,  np.int32)
        area3 = np.array(self.area3,  np.int32)
        area4 = np.array(self.area4,  np.int32)
        #area5 = np.array(self.area5,  np.int32)
        area6 = np.array(self.area6,  np.int32)
        area7 = np.array(self.area7,  np.int32)
        area8 = np.array(self.area8,  np.int32)

        mask = np.dstack((mask, mask, mask))

        cv2.fillPoly(mask, [area1], (0,255,255))
        cv2.fillPoly(mask, [area2], (0,255,0))
        cv2.fillPoly(mask, [area3], (255,255,0))
        cv2.fillPoly(mask, [area4], (150,30,30))
        #cv2.fillPoly(mask, [area5], (0,255,150))
        cv2.fillPoly(mask, [area6], (0,50,150))
        cv2.fillPoly(mask, [area7], (255,50,150))
        cv2.fillPoly(mask, [area8], (50,10,150))


        mask2 = cv2.bitwise_or(image, mask)
        return mask2

    def area_check(self, x, y, id):
        point=(int(x),int(y))
        #1
        if self.area1_path.contains_point(point):
            ID_FLAG = self.id_check(self.id_list1, id)
            if ID_FLAG==False:
                #print("alan1")
                if self.in_out==[]:
                    self.in_out.append([str(id), '1', '0'])

                else:
                    for i, row in enumerate(self.in_out):
                        if str(id) == row[0]:
                            self.in_out[i][2] = '1'

                            _, inn, outt = self.in_out[i]
                            self.main_matrix[int(inn)-1][int(outt)-1] += 1
                            del self.in_out[i]
                            #
                            #self.in_out.append([str(id), '1', '0'])
                            #
                            self.flag1 = True

                    #if self.flag1==False:
                    self.in_out.append([str(id),'1','0'])

                    self.flag1 = False

                self.id_list1 = np.append(self.id_list1,id)
        #2
        elif self.area2_path.contains_point(point):
            ID_FLAG = self.id_check(self.id_list2, id)
            if ID_FLAG==False:
                #print("alan2")
                if self.in_out==[]:
                    self.in_out.append([str(id), '2', '0'])

                else:
                    for i, row in enumerate(self.in_out):
                        if str(id) == row[0]:
                            self.in_out[i][2] = '2'
                
                            _, inn, outt = self.in_out[i]
                            self.main_matrix[int(inn)-1][int(outt)-1] += 1
                            del self.in_out[i]
                            #
                            #self.in_out.append([str(id), '2', '0'])
                            #
                            self.flag1 = True

                    #if self.flag1==False:
                    self.in_out.append([str(id),'2','0'])

                    self.flag1 = False

                self.id_list2 = np.append(self.id_list2,id)
        #3
        elif self.area3_path.contains_point(point):
            ID_FLAG = self.id_check(self.id_list3, id)
            if ID_FLAG==False:
                # print("alan2")
                if self.in_out==[]:
                    self.in_out.append([str(id), '3', '0'])

                else:
                    for i, row in enumerate(self.in_out):
                        if str(id) == row[0]:
                            self.in_out[i][2] = '3'
                
                            _, inn, outt = self.in_out[i]
                            self.main_matrix[int(inn)-1][int(outt)-1] += 1
                            del self.in_out[i]

                            self.flag1 = True

                    #if self.flag1==False:
                    self.in_out.append([str(id),'3','0'])

                    self.flag1 = False

                self.id_list3 = np.append(self.id_list3,id)
        #4
        elif self.area4_path.contains_point(point):
            ID_FLAG = self.id_check(self.id_list4, id)
            if ID_FLAG==False:
                # print("alan2")
                if self.in_out==[]:
                    self.in_out.append([str(id), '4', '0'])

                else:
                    for i, row in enumerate(self.in_out):
                        if str(id) == row[0]:
                            self.in_out[i][2] = '4'
                
                            _, inn, outt = self.in_out[i]
                            self.main_matrix[int(inn)-1][int(outt)-1] += 1
                            del self.in_out[i]

                            self.flag1 = True

                    #if self.flag1==False:
                    self.in_out.append([str(id),'4','0'])

                    self.flag1 = False

                self.id_list4 = np.append(self.id_list4,id)
        ##5
        # elif self.area5_path.contains_point(point):
        #     ID_FLAG = self.id_check(self.id_list5, id)
        #     if ID_FLAG==False:
        # #         print("alan2")
        #         if self.in_out==[]:
        #             self.in_out.append([str(id), '5', '0'])

        #         else:
        #             for i, row in enumerate(self.in_out):
        #                 if str(id) == row[0]:
        #                     self.in_out[i][2] = '5'
                
        #                     _, inn, outt = self.in_out[i]
        #                     self.main_matrix[int(inn)-1][int(outt)-1] += 1
        #                     del self.in_out[i]

        #                     self.flag1 = True

        #             #if self.flag1==False:
        #             self.in_out.append([str(id),'5','0'])

        #             self.flag1 = False

        #         self.id_list5 = np.append(self.id_list5,id)
        #6
        elif self.area6_path.contains_point(point):
            ID_FLAG = self.id_check(self.id_list6, id)
            if ID_FLAG==False:
                # print("alan2")
                if self.in_out==[]:
                    self.in_out.append([str(id), '6', '0'])

                else:
                    for i, row in enumerate(self.in_out):
                        if str(id) == row[0]:
                            self.in_out[i][2] = '6'
                
                            _, inn, outt = self.in_out[i]
                            self.main_matrix[int(inn)-1][int(outt)-1] += 1
                            del self.in_out[i]
                            #
                            #self.in_out.append([str(id), '6', '0'])
                            #
                            self.flag1 = True

                    #if self.flag1==False:
                    self.in_out.append([str(id),'6','0'])

                    self.flag1 = False

                self.id_list6 = np.append(self.id_list6,id)
        #7
        elif self.area7_path.contains_point(point):
            ID_FLAG = self.id_check(self.id_list7, id)
            if ID_FLAG==False:
                # print("alan2")
                if self.in_out==[]:
                    self.in_out.append([str(id), '7', '0'])

                else:
                    for i, row in enumerate(self.in_out):
                        if str(id) == row[0]:
                            self.in_out[i][2] = '7'
                
                            _, inn, outt = self.in_out[i]
                            self.main_matrix[int(inn)-1][int(outt)-1] += 1
                            del self.in_out[i]

                            self.flag1 = True

                    #if self.flag1==False:
                    self.in_out.append([str(id),'7','0'])

                    self.flag1 = False

                self.id_list7 = np.append(self.id_list7,id)
        #8
        elif self.area8_path.contains_point(point):
            ID_FLAG = self.id_check(self.id_list8, id)
            if ID_FLAG==False:
                # print("alan2")
                if self.in_out==[]:
                    self.in_out.append([str(id), '8', '0'])

                else:
                    for i, row in enumerate(self.in_out):
                        if str(id) == row[0]:
                            self.in_out[i][2] = '8'
                
                            _, inn, outt = self.in_out[i]
                            self.main_matrix[int(inn)-1][int(outt)-1] += 1
                            del self.in_out[i]

                            self.flag1 = True

                    #if self.flag1==False:
                    self.in_out.append([str(id),'8','0'])

                    self.flag1 = False

                self.id_list8 = np.append(self.id_list8,id)

        else:
            pass
        
    def id_check(self, id_list, current_id):
        if current_id in id_list:
            self.id_flag = True
        else:
            self.id_flag = False
        return self.id_flag

if __name__ == "__main__":
    DC = Detector()
    VC = Vehicle_Count()
    cap = cv2.VideoCapture("enter_video_path")
    while (cap.isOpened()):
        ret, image_rgb  = cap.read()
        #image_rgb=cv2.resize(image_rgb,(1280,960),interpolation=cv2.INTER_AREA)
        dtimg, dtt = DC.main_obj_counting(image_rgb)
        resimg = VC.gui(dtt, dtimg)

        #cv2.resize(resimg, (1289,728))
        cv2.imshow("Inference", resimg)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

