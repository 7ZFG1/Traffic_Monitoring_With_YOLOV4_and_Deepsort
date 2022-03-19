import enum
import random

import darknet
import time
import cv2
import threading
import glob
import os
import matplotlib.path as mplPath
#from object_counting import Count
import numpy as np
#from my_deepsort import Tracking


class Detector:
    def __init__(self):
        #self.fm_match = fm_Matcher.featureMatch() 
        #self.counter = Count()
        #self.tracking = Tracking()
        self.icons =       { "vehicle": "icons/car.png", 
                            "pedestrian": "icons/pedestrian.png",
                            }
        self.color_mine = { "vehicle": (0,255,255), 
                            "person": (0,255,0),
                            "car": (50,255,255),
                            "truck": (100,100,255),
                            "van": (180,130,70),
                            "bus": (50,255,150),
                            "motorbike": (130,0,75)
                            }
        self.detections = []
        self.thresh = 0.3
        config_file = "cfg/yolov4.cfg"
        data_file = "data/coco.data"
        weight_file = "yolov4.weights"
        
        # config_file = "cfg/yolov4-custom.cfg"
        # data_file = "data/object.data"
        # #weight_file = "data/yolov4-my_best_1.weights"
        
        # weight_file = "backup/yolov4-custom_last.weights"

        ##
        ##

        batch_size = 1
        self.boxed_image = []
        random.seed(3)  # deterministic bbox colors
        self.network, self.class_names, self.class_colors = darknet.load_network(
            config_file, data_file, weight_file, batch_size=batch_size
        )
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)

    def image_detection(self, image):
        myimg = image.copy()
        myimg = cv2.resize(myimg, (self.width, self.height))
        # Darknet doesn't accept numpy images.
        # Create one with image we reuse for each detect
        prev_time = time.time()
    
        darknet_image = darknet.make_image(self.width, self.height, 3)
        image_resized = cv2.resize(image, (self.width, self.height))
        org = image_resized

        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        self.detections = darknet.detect_image(
            self.network, self.class_names, darknet_image, thresh=self.thresh
        )
        darknet.free_image(darknet_image)
        image, my_detections = darknet.draw_boxes(self.detections, image_resized, self.class_colors)
        #
        #my_detections=self.feature_matching(my_detections)
        #
        #darknet.print_detections(detections)
        
        #self.boxed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.boxed_image = image
        self.fps = int(1 / (time.time() - prev_time))
        #print("FPS: {}".format(fps))
        my_detections=self.out_matching_bbox(my_detections)
        return my_detections, self.boxed_image, myimg

    def fix_det_list(self, detection):
        tmp_detection = detection
        for i,pred in enumerate(detection):
            dots_list=[]
            self.xmin,self.min,self.xmax,self.ymax = self.bbox(pred)
            path = [[self.xmin,self.ymin],[self.xmax,self.ymin],[self.xmin,self.ymax],[self.xmax,self.ymax]]
            #
            self.area_path = mplPath.Path(np.array(path))
            for i, dots in enumerate(self.mid_points):
                if self.area_path.contains_point(dots):
                    dots_list = np.append(dots_list, i)
            
            if len(dots_list)==2:
                if float(self.conf[int(dots_list[0])]) > float(self.conf[int(dots_list[1])]):
                    tmp_detection[int(dots_list[1])][2]=1,1,1,1
                else:
                    tmp_detection[int(dots_list[0])][2]=1,1,1,1
        return tmp_detection

    def mid_dots(self, pred):
        bbox_width = (pred[2][2]*self.w)/608    ##416---608
        bbox_height = (pred[2][3]*self.h)/608    ##416---608
        center_x = (pred[2][0]*self.w)/608    ##416---608
        center_y = (pred[2][1]*self.h)/608    ##416---608
        self.xmin = int(center_x - (bbox_width / 2))
        self.ymin = int(center_y - (bbox_height / 2))
        self.xmax = int(center_x + (bbox_width / 2))
        self.ymax = int(center_y + (bbox_height / 2))

        self.xmid = int((self.xmax+self.xmin)/2)
        self.ymid = int((self.ymax+self.ymin)/2)
        
        return self.xmid, self.ymid

    def bbox(self, pred):
        bbox_width = (pred[2][2]*self.w)/608    ##416---608
        bbox_height = (pred[2][3]*self.h)/608    ##416---608
        center_x = (pred[2][0]*self.w)/608    ##416---608
        center_y = (pred[2][1]*self.h)/608    ##416---608
        self.xmin = int(center_x - (bbox_width / 2))
        self.ymin = int(center_y - (bbox_height / 2))
        self.xmax = int(center_x + (bbox_width / 2))
        self.ymax = int(center_y + (bbox_height / 2))
        return self.xmin,self.ymin,self.xmax,self.ymax
    def calc_bbox(self, pred):
        bbox_width = pred[2][2]
        bbox_height = pred[2][3]
        center_x = pred[2][0]
        center_y = pred[2][1]
        self.xmin = int(center_x - (bbox_width / 2))
        self.ymin = int(center_y - (bbox_height / 2))
        self.xmax = int(center_x + (bbox_width / 2))
        self.ymax = int(center_y + (bbox_height / 2))
        return self.xmin,self.ymin,self.xmax,self.ymax

    def run_gui(self):
        threading.Thread(target=self.gui).start()

    def gui(self):
        while True:
            time.sleep(0.02)
            if self.boxed_image != []:
                cv2.imshow("Inference", cv2.resize(self.boxed_image,(812,812),interpolation=cv2.INTER_AREA))
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

    def main(self, image_rgb):
        self.h, self.w, d = image_rgb .shape
        detection, img, myimg = self.image_detection(image_rgb)
        #
        # self.mid_points = [self.mid_dots(pred) for i,pred in enumerate(detection) if detection!=[]]
        # self.conf = [pred[1] for i,pred in enumerate(detection) if detection!=[]]
        # detection = self.fix_det_list(detection)
        #

        if detection != []:
            for i,pred in enumerate(detection):
                self.xmin,self.min,self.xmax,self.ymax = self.bbox(pred)
                # path = [[self.xmin,self.ymin],[self.xmax,self.ymin],[self.xmin,self.ymax],[self.xmax,self.ymax]]
                # #
                # self.area_path = mplPath.Path(np.array(path))
                # for i, dots in enumerate(self.mid_points):
                #     if self.area_path.contains_point(dots):
                #         dots_list = np.append(dots_list, i)
                
                # if len(dots_list)==2:
                #     if float(self.conf[dots_list[0]]) > float(self.conf[dots_list[1]]):
                #         tmp_detection[int(dots_list[1])][2]=1,1,1,1
                #     else:
                #         tmp_detection[int(dots_list[0])][2]=1,1,1,1
                #
                image_rgb = cv2.rectangle(image_rgb, (self.xmin, self.ymin), (self.xmax, self.ymax), self.color_mine[pred[0]], 1)
                image_rgb = cv2.rectangle(image_rgb, (self.xmin, self.ymin-20), (self.xmax, self.ymin), self.color_mine[pred[0]], -1)  #147,112,219
                cv2.putText(image_rgb, pred[0]+" "+pred[1], (self.xmin,self.ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA) #with acc
                cv2.putText(image_rgb, 'FPS: '+str(self.fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
                image_rgb = cv2.rectangle(image_rgb, (1200,60), (1730, 0), (255,255,255), -1)
                cv2.putText(image_rgb, 'Object Detection', (1200,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
        return image_rgb, detection

    def main_intensity(self, image_rgb):
        self.h, self.w, d = image_rgb .shape
        detection, img, myimg = self.image_detection(image_rgb)
        black = np.zeros((960,1280,3),dtype='uint8')
        if detection != []:
            for i,pred in enumerate(detection):
                self.xmin,self.min,self.xmax,self.ymax = self.bbox(pred)
                image_rgb = cv2.rectangle(image_rgb, (self.xmin, self.ymin), (self.xmax, self.ymax), self.color_mine[pred[0]], 1)
                image_rgb = cv2.rectangle(image_rgb, (self.xmin, self.ymin-20), (self.xmax, self.ymin), self.color_mine[pred[0]], -1)  #147,112,219
                cv2.putText(image_rgb, pred[0]+" "+pred[1], (self.xmin,self.ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA) #with acc
                cv2.putText(image_rgb, 'FPS: '+str(self.fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)

                
                black = cv2.rectangle(black, (self.xmin, self.ymin), (self.xmax, self.ymax), self.color_mine[pred[0]], -1)
        return image_rgb, detection, black

    def main_obj_counting(self, image_rgb):
        self.h, self.w, d = image_rgb .shape
        detection, img, myimg = self.image_detection(image_rgb)

        if detection != []:
            for i,pred in enumerate(detection):
                self.xmin,self.min,self.xmax,self.ymax = self.bbox(pred)
                image_rgb = cv2.rectangle(image_rgb, (self.xmin, self.ymin), (self.xmax, self.ymax), self.color_mine[pred[0]], 1)
                #image_rgb = cv2.rectangle(image_rgb, (self.xmin, self.ymin-20), (self.xmax, self.ymin), self.color_mine[pred[0]], -1)  #147,112,219
                #cv2.putText(image_rgb, pred[0]+" "+pred[1], (self.xmin,self.ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA) #with acc
                #cv2.putText(image_rgb, pred[0], (self.xmin,self.ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        return image_rgb, detection

    def main_deepsort(self, image_rgb):
        self.h, self.w, d = image_rgb .shape
        detection, img, myimg = self.image_detection(image_rgb)
        return detection

    def out_matching_bbox(self,detection):
        tmp_idx = []
        delete_list = []
        tmp_detection=detection
        bboxes = [self.bbox(pred) for i,pred in enumerate(detection) if detection!=[]]
        firstcx2 = [predc[2] for i,predc in enumerate(bboxes)]
        firstcy1 = [predc[1] for i,predc in enumerate(bboxes)]
        #
        firstcx1 = [predc[0] for i,predc in enumerate(bboxes)]
        firstcy2 = [predc[3] for i,predc in enumerate(bboxes)]
        #
        for i, pred in enumerate(detection):
            x1,y1,x2,y2 = self.bbox(pred)
    
            firstcx2 = np.array(firstcx2)
            firstcy1 = np.array(firstcy1)
            #
            firstcx1 = np.array(firstcx1)
            firstcy2 = np.array(firstcy2)

            index5=np.where(firstcx1<=x1+100)
            index6=np.where(firstcx1>=x1-100)

            index7=np.where(firstcy2<=y2+100)
            index8=np.where(firstcy2>=y2-100)

            res3=np.intersect1d(index5, index6)
            res4=np.intersect1d(index7, index8)
            index_1=np.intersect1d(res3, res4)
            #
            index1=np.where(firstcx2<=x2+100)
            index2=np.where(firstcx2>=x2-100)

            index3=np.where(firstcy1<=y1+100)
            index4=np.where(firstcy1>=y1-100)

            res1=np.intersect1d(index1, index2)
            res2=np.intersect1d(index3, index4)
            index_2 = np.intersect1d(res1, res2)

            index = np.intersect1d(index_2, index_1)
            if len(index)>1:
                first_idx = index[0]
                second_idx = index[1]
                if not (second_idx and first_idx) in tmp_idx: 
                    name1 = detection[first_idx][0]
                    name2 = detection[second_idx][0]
                    if name1!=name2:
                        tmp_idx = np.append(tmp_idx,first_idx)

                        first_acc = detection[first_idx][1]
                        seconde_acc = detection[second_idx][1]
                        if first_acc>=seconde_acc:
                            #tmp_detection.pop(second_idx)
                            delete_list = np.append(delete_list, second_idx)
                        else:
                            #tmp_detection.pop(first_idx)
                            delete_list = np.append(delete_list, first_idx)

        for i in range(len(delete_list)):
            tmp_detection[int(delete_list[i])][2]=1,1,1,1
            # tmp_detection[int(delete_list[i])][0]=' '
            # tmp_detection[int(delete_list[i])][1]=' '
        return tmp_detection

    def feature_matching(self, detection):
        for i,pred in enumerate(detection):
            name = pred[0]
            if name=='truck' or name=='van' or 'bus':
                x1,y1,x2,y2 = self.bbox(pred)
                img = image_rgb[y1:y2,x1:x2]

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                low_green = np.array([36, 25, 25])
                high_green = np.array([70, 255,255])

                green_mask = cv2.inRange(hsv, low_green, high_green)
                mean = np.mean(green_mask==255)
                if mean>0.2:
                    pass
                    # detection[i][0]='bus'
                    # print(detection)
                else:
                    img2 = image_rgb[y1:y2,x1:x2]
                    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    mean2 = np.mean(gray>=80)
                    if mean2>0.4:
                        pass
                        #detection[i][0]='van'
                    else:
                        detection[i][0]='truck'
        return detection
 

            
        #self.fm_match.check_vehicle(img2,'truck')

if __name__ == "__main__":
    DC = Detector()

    cap = cv2.VideoCapture("enter_your_video_path")
    cnt=0
    while (cap.isOpened()):
        #time.sleep(0.04)
        ret, image_rgb  = cap.read()
        #image_rgb=cv2.resize(image_rgb,(1280,960),interpolation=cv2.INTER_AREA)
        dtimg, dtt = DC.main(image_rgb)
        #out.write(dtimg)
        dtimg=cv2.resize(dtimg,(1290,960),interpolation=cv2.INTER_AREA)
        cv2.imshow("Inference", dtimg)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break





  
