import cv2
import time
import os
import glob
import numpy as np
from detection import Detector
from my_deepsort import Tracking

class Count:
    def __init__(self):
        self.tracking = Tracking()
        self.id_flag = False
        self.id_list_left = []
        self.id_list_right = []
        self.y_thresh_left = 780 
        self.y_thresh_right = 780
        self.x_thresh = 720
        self.cnt_car_l = 0
        self.cnt_truck_l = 0
        self.cnt_van_l = 0
        self.cnt_bus_l = 0
        self.cnt_motorbike_l = 0
        self.cnt_car_r = 0
        self.cnt_truck_r = 0
        self.cnt_van_r = 0
        self.cnt_bus_r = 0
        self.cnt_motorbike_r = 0 
        self.points_left = [[678.1,641.5],[679.1,559.3],[467.2,556.3],[465.3,639.5]]        #Derinceuzak
        self.points_right = [[716.8,639.5],[719.7,558.3],[911.8,564.3],[907.8,643.5]]

    def gui(self, detections, image):
        self.h, self.w, d = image.shape
        self.image=self.mask_img((255,0,0),image)
        img = self.run_gui2(detections)
        return img
                
    def run_gui2(self, detections):  
        tr_bbox, ids, pred = self.tracking.run_gui4out(image_rgb, detections)  
        if tr_bbox!=[]: 
            self.idx_list = []
            self.flag=True 
            indexs_list=self.match_name_start(tr_bbox, detections)
            for i, bx in enumerate(tr_bbox[:len(pred)]):
                xmin=bx[0]
                ymin=bx[1]
                xmax=bx[2]
                ymax=bx[3]
                if xmin<690 and ymin>595: 
                    ID_FLAG = self.id_check(self.id_list_left, ids[i])
                    if ID_FLAG==False:
                        self.id_list_left = np.append(self.id_list_left,ids[i])
                        dt_index=self.match_name(indexs_list,i)
                        if dt_index!=None:
                            if detections[dt_index][0]=="car":
                                self.cnt_car_l+=1
                                self.image=cv2.putText(self.image, 'car: '+str(self.cnt_car_l), (485,580), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                            elif detections[dt_index][0]=='truck':
                                self.cnt_truck_l+=1
                                self.image=cv2.putText(self.image, 'truck: '+str(self.cnt_truck_l), (485,605), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                            elif detections[dt_index][0]=='van':
                                self.cnt_van_l+=1
                                self.image=cv2.putText(self.image, 'van: '+str(self.cnt_van_l), (485,625), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                            elif detections[dt_index][0]=='bus':
                                self.cnt_bus_l+=1
                                self.image=cv2.putText(self.image, 'bus:'+str(self.cnt_bus_l), (580,590), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                            elif detections[dt_index][0]=='motorbike':
                                self.cnt_motorbike_l+=1
                                self.image=cv2.putText(self.image, 'motor:'+str(self.cnt_motorbike_l), (580,630), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                            else:
                                print("name matching failed on left side")

                elif xmin>690 and ymin>595 and ymin<640:
                    ID_FLAG = self.id_check(self.id_list_right, ids[i])
                    if ID_FLAG==False:
                        self.id_list_right = np.append(self.id_list_right,ids[i])
                        dt_index=self.match_name(indexs_list,i)
                        if dt_index!=None:
                            if detections[dt_index][0]=="car":
                                self.cnt_car_r+=1
                                self.image=cv2.putText(self.image, 'car: '+str(self.cnt_car_r), (720,580), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                            elif detections[dt_index][0]=='truck':
                                self.cnt_truck_r+=1
                                self.image=cv2.putText(self.image, 'truck: '+str(self.cnt_truck_r), (720,605), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                            elif detections[dt_index][0]=='van':
                                self.cnt_van_r+=1
                                self.image=cv2.putText(self.image, 'van: '+str(self.cnt_van_r), (720,625), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                            elif detections[dt_index][0]=='bus':
                                self.cnt_bus_r+=1
                                self.image=cv2.putText(self.image, 'bus:'+str(self.cnt_bus_r), (815,590), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                            elif detections[dt_index][0]=='motorbike':
                                self.cnt_motorbike_r+=1
                                self.image=cv2.putText(self.image, 'motor:'+str(self.cnt_motorbike_r), (815,630), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                            else:
                                print("name matching failed on right side")

                else:
                    self.image=cv2.putText(self.image, 'car: '+str(self.cnt_car_l), (485,580), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    self.image=cv2.putText(self.image, 'truck: '+str(self.cnt_truck_l), (485,605), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    self.image=cv2.putText(self.image, 'van: '+str(self.cnt_van_l), (485,625), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    self.image=cv2.putText(self.image, 'bus: '+str(self.cnt_bus_l), (580,590), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    self.image=cv2.putText(self.image, 'motor: '+str(self.cnt_motorbike_l), (580,630), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)

                    self.image=cv2.putText(self.image, 'car: '+str(self.cnt_car_r), (720,580), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    self.image=cv2.putText(self.image, 'truck: '+str(self.cnt_truck_r), (720,605), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    self.image=cv2.putText(self.image, 'van: '+str(self.cnt_van_r), (720,625), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    self.image=cv2.putText(self.image, 'bus:'+str(self.cnt_bus_r), (815,590), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                    self.image=cv2.putText(self.image, 'motor:'+str(self.cnt_motorbike_r), (815,630), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        return self.image

    def calc_bbox(self, pred):
        bbox_width = (pred[2][2]*self.w)/416
        bbox_height = (pred[2][3]*self.h)/416
        center_x = (pred[2][0]*self.w)/416
        center_y = (pred[2][1]*self.h)/416
        self.xmin = int(center_x - (bbox_width / 2))
        self.ymin = int(center_y - (bbox_height / 2))
        self.xmax = int(center_x + (bbox_width / 2))
        self.ymax = int(center_y + (bbox_height / 2))
        return self.xmin, self.ymin, self.xmax, self.ymax

    def mask_img(self, color, image):
        h, w, d = image.shape
        mask = np.zeros([h,w],dtype=np.uint8)

        pts_left = np.array(self.points_left,  np.int32)
        pts_right = np.array(self.points_right,  np.int32)

        mask = np.dstack((mask, mask, mask))

        cv2.fillPoly(mask, [pts_left], (255,0,0))
        cv2.fillPoly(mask, [pts_right], (0,0,255))

        mask = cv2.bitwise_or(image, mask)
        return mask

    def id_check(self, id_list, current_id):
        if current_id in id_list:
            self.id_flag = True
        else:
            self.id_flag = False
        return self.id_flag

    def match_name_start(self, trbbx, detect):
        firstcx = [predc[0] for i,predc in enumerate(trbbx)]
        firstcy = [predc[1] for i,predc in enumerate(trbbx)]
        for i, pred in enumerate(detect):
            x1,y1,x2,y2 = self.calc_bbox(pred)

            firstcx = np.array(firstcx)
            firstcy = np.array(firstcy)

            index1=np.where(firstcx<=x1+8)
            index2=np.where(firstcx>=x1-8)

            index3=np.where(firstcy<=y1+8)
            index4=np.where(firstcy>=y1-8)

            res1=np.intersect1d(index1, index2)
            res2=np.intersect1d(index3, index4)
            index = np.intersect1d(res1, res2)
            try:
                if self.flag:
                    self.idx_list = "{} {} {}".format(index[0],",",i)
                    self.flag=False
                else:
                    idx_list2 = "{} {} {}".format(index[0],",",i)
                    self.idx_list = np.append(self.idx_list, idx_list2)
            except:
                print("array is empty")
        return self.idx_list

    def match_name(self, idx_list, tr_idx):
        for i in range(len(idx_list)):
            first = idx_list[i].split(',')
            if int(first[0])==tr_idx:
                detect_idx=int(first[1])
                break
            else:
                detect_idx=None
        return detect_idx

if __name__ == "__main__":
    counter = Count()
    DC = Detector()
    # cap = cv2.VideoCapture("enter_video_path")
    # while (cap.isOpened()):
    #     ret, image_rgb  = cap.read()
    #     image_rgb=cv2.resize(image_rgb,(1280,960),interpolation=cv2.INTER_AREA)
    #     dtimg, dtt = DC.main_obj_counting(image_rgb)
    #     cntimg = counter.gui(dtt, dtimg)
    #     #cv2.imwrite("/home/zfg/Desktop/derince.jpg", cntimg)

    #     cv2.imshow("Inference", cntimg)
    #     if cv2.waitKey(10) & 0xFF == ord("q"):
    #         break

    for fname in sorted(glob.glob("enter_image_path/*.jpg"), key=os.path.getmtime): 
        time.sleep(0.01)
        image_rgb  = cv2.imread(fname)
        image_rgb=cv2.resize(image_rgb,(1280,960),interpolation=cv2.INTER_AREA)
        dtimg, dtt = DC.main_obj_counting(image_rgb)
        cntimg = counter.gui(dtt, dtimg)

        cntimg=cv2.resize(cntimg,(2592,1520),interpolation=cv2.INTER_AREA)
        cntimg = cv2.rectangle(cntimg, (1200,60), (1730, 0), (255,255,255), -1)
        cv2.putText(cntimg, 'Object Counting', (1200,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
        #cv2.imwrite("/home/zfg/Desktop/derince.jpg", cntimg)

        cv2.imshow("Inference", cntimg)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break