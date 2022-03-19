import os
import cv2
import time
import glob
import numpy as np
import random
print("[INFO] Loading...")
from deep_sort import DeepSort
import matplotlib.path as mplPath
#from object_counting import Count
#from detection import Detector

class Tracking:
    def __init__(self):
        self.outputs = []
        self.identities = []
        #self.DC = Detector()
        #self.counter = Count()
        self.deepsort_checkpoint = "checkpoint/ckpt.t7"
        self.deepsort = DeepSort(self.deepsort_checkpoint)
        self.name = []
        self.conf = []
        self.bbox = []
        self.colors =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]

        self.area1 = [[846,645],[408,586],[257,696],[809,751]]        
        self.area2 = [[82,380],[388,428],[376,485],[46,414]]

        self.area1_path = mplPath.Path(np.array(self.area1))
        self.area2_path = mplPath.Path(np.array(self.area2))
        
    
    def gui(self):
        self.DC = Detector()
        for fname in sorted(glob.glob("enter_images_path/*.jpg"), key=os.path.getmtime):         
            s = time.time()
            time.sleep(0.01)
            self.image_rgb  = cv2.imread(fname)
            self.h, self.w, d = self.image_rgb.shape
            self.detections = self.DC.main_deepsort(self.image_rgb)  
            if self.detections != []:
                self.run_gui()

            self.fps = int(1 / (time.time() - s))
            #cv2.imshow("detection",self.detectimg)

            self.image_rgb = cv2.rectangle(self.image_rgb, (1200,60), (1730, 0), (255,255,255), -1)
            cv2.putText(self.image_rgb, 'Object Tracking', (1200,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(self.image_rgb, 'FPS: '+str(self.fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
            cv2.imshow("tracking",cv2.resize(self.image_rgb,(int(self.w/2),int(self.h/2))))
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
            
            print("FPS: {}".format(self.fps))

    def gui_video(self):

        self.DC = Detector()
        cap = cv2.VideoCapture("enter_video_path")
        while (cap.isOpened()):
            s = time.time()
            ret, self.image_rgb  = cap.read()

            self.h, self.w, d = self.image_rgb.shape

            #self.detections, self.detectimg, self.trimg = self.DC.image_detection(self.image_rgb) 
            self.detections = self.DC.main_deepsort(self.image_rgb) 
            if self.detections != []:
                self.run_gui()

            self.fps = int(1 / (time.time() - s))
            #cv2.imshow("detection",self.detectimg)

            self.image_rgb = cv2.rectangle(self.image_rgb, (1200,60), (1730, 0), (255,255,255), -1)
            cv2.putText(self.image_rgb, 'Object Tracking', (1200,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(self.image_rgb, 'FPS: '+str(self.fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
            #cv2.imwrite("/home/zfg/Desktop/saveddd.jpg",self.image_rgb)
            #self.image_rgb=cv2.resize(self.image_rgb,(int(self.w/2),int(self.h/2)))
            #self.image_rgb = self.mask_img(self.image_rgb)
            #
            # if len(self.outputs) > 0:
            #     bbox_xyxy = self.outputs[:, :4]
            #     identities = self.outputs[:, -1]
            #     self.identities = identities
            #     bbox1 = [self.tlwh_to_xywh(psebbox) for i,psebbox in enumerate(bbox_xyxy) if bbox_xyxy!=[]]
            #     bbox1 = [self.calc_bbox(psebbox) for i,psebbox in enumerate(bbox1) if bbox1!=[]]
            #     self.area_check(bbox1, identities)
            #
            cv2.imshow("tracking",self.image_rgb)
            #cv2.imwrite("/home/zfg/Desktop/TRACK2.jpg",cv2.resize(self.image_rgb,(int(self.w/2),int(self.h/2))))
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
            
            #print("FPS: {}".format(self.fps))

    def run_gui4out(self, img, detection):
        self.h, self.w, d = img.shape
        self.name = []
        self.conf = []
        self.bbox = []
        self.name = [pred[0] for i,pred in enumerate(detection) if detection!=[]]
        self.conf = [pred[1] for i,pred in enumerate(detection) if detection!=[]]
        self.bbox = [pred[2] for i,pred in enumerate(detection) if detection!=[]]

        outputs = self.deepsort.update(self.bbox, self.conf, img)
        bbox=[]
        identities=[]
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]

            bbox = [self.tlwh_to_xywh(psebbox) for i,psebbox in enumerate(bbox_xyxy) if bbox_xyxy!=[]]
            bbox = [self.calc_bbox(psebbox) for i,psebbox in enumerate(bbox) if bbox!=[]]

        return bbox, identities, self.name

    def run_gui(self):
        self.name = []
        self.conf = []
        self.bbox = []
        self.name = [pred[0] for i,pred in enumerate(self.detections) if self.detections!=[]]
        self.conf = [pred[1] for i,pred in enumerate(self.detections) if self.detections!=[]]
        self.bbox = [pred[2] for i,pred in enumerate(self.detections) if self.detections!=[]]
        #self.bbox = [self.calc_bbox(pred[2]) for i,pred in enumerate(self.detections) if self.detections!=[]]
        # for i in range(len(self.bbox)):
        #     self.bbox[i]=np.append(self.bbox[i],self.name[i])


        self.outputs = self.deepsort.update(self.bbox, self.conf, self.image_rgb)
 
        if len(self.outputs) > 0:
            bbox_xyxy = self.outputs[:, :4]
            identities = self.outputs[:, -1]
            self.identities = identities
            bbox1 = [self.tlwh_to_xywh(psebbox) for i,psebbox in enumerate(bbox_xyxy) if bbox_xyxy!=[]]
            bbox1 = [self.calc_bbox(psebbox) for i,psebbox in enumerate(bbox1) if bbox1!=[]]
            #
            #bbox_xyxy2 = [self.calc_bbox(psebbox) for i,psebbox in enumerate(bbox_xyxy) if bbox_xyxy!=[]]
            #
            #self.trimg = self.draw_bbox(self.trimg, bbox_xyxy, identities)
            #self.area_check(bbox1, identities)
            self.image_rgb = self.draw_bbox(self.image_rgb, bbox_xyxy, identities)
        

    def plot_one_box(self, x, ori_img, color=None, label=None, line_thickness=None):
        img = ori_img
        tl = line_thickness or round(
            0.002 * max(img.shape[0:2])) + 1 
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        #cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            # cv2.putText(img,
            #             label, (c1[0], c1[1] - 2),
            #             0,
            #             tl / 3, [225, 255, 255],
            #             thickness=tf,
            #             lineType=cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        return img

    def draw_bboxes(self, ori_img, bbox, identities=None, offset=(0,0)):
        img = ori_img
        for i,box in enumerate(bbox[:len(identities)]):
            x1,y1,x2,y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = self.colors[id%len(self.colors)]
            label = '{}{:d}'.format("", id)
            #
            #t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
            #
            img = self.plot_one_box([x1,y1,x2,y2], img, color, label)
            #
            # cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
            # cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
            # cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
            #
        return img

    def draw_bbox(self, img, bbox, ids):
        bbox = [self.tlwh_to_xywh(psebbox) for i,psebbox in enumerate(bbox) if bbox!=[]]
        bbox = [self.calc_bbox(psebbox) for i,psebbox in enumerate(bbox) if bbox!=[]]
        for i,box in enumerate(bbox[:len(ids)]):
            x1,y1,x2,y2 = [int(i) for i in box]
            id = int(ids[i]) if ids is not None else 0
            color = self.colors[id%len(self.colors)]
            #cv2.circle(img, (x1+15,y1-30), 15, color, -1)
            img = cv2.rectangle(img, (x1,y1), (x2, y2), color, 2)
            if len(str(id))==1:
                img = cv2.rectangle(img, (x1, y1-17), (x1+15, y1-41), color, -1)
                cv2.putText(img, str(id), (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
            elif len(str(id))==2:
                img = cv2.rectangle(img, (x1, y1-17), (x1+30, y1-41), color, -1)
                cv2.putText(img, str(id), (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
            else:
                img = cv2.rectangle(img, (x1, y1-17), (x1+45, y1-41), color, -1)
                cv2.putText(img, str(id), (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
            
        return img

    def calc_bbox(self, pred):
        bbox_width = (pred[2]*self.w)/608
        bbox_height = (pred[3]*self.h)/608
        center_x = (pred[0]*self.w)/608
        center_y = (pred[1]*self.h)/608
        self.xmin = int(center_x - (bbox_width / 2))
        self.ymin = int(center_y - (bbox_height / 2))
        self.xmax = int(center_x + (bbox_width / 2))
        self.ymax = int(center_y + (bbox_height / 2))
        return self.xmin,self.ymin,self.xmax,self.ymax
        
    def tlwh_to_xywh(self, bbox_xywh):
        a1 = bbox_xywh[0] + bbox_xywh[2]/2
        a2 = bbox_xywh[1] + bbox_xywh[3]/2
        a3 = bbox_xywh[2]
        a4 = bbox_xywh[3]
        return a1,a2,a3,a4

    def in_out(self):
        pass

    def mask_img(self, image):
        h, w, d = image.shape
        mask = np.zeros([h,w],dtype=np.uint8)

        area1 = np.array(self.area1,  np.int32)
        area2 = np.array(self.area2,  np.int32)

        mask = np.dstack((mask, mask, mask))

        cv2.fillPoly(mask, [area1], (0,255,255))
        cv2.fillPoly(mask, [area2], (0,255,0))

        mask2 = cv2.bitwise_or(image, mask)
        return mask2

    def area_check(self, bbox, id):
        for i,box in enumerate(bbox):
            _,_,x,y = box
            point=(int(x),int(y))

            if self.area1_path.contains_point(point):
                print("alan1")
            elif self.area2_path.contains_point(point):
                print("alan2")
            else:
                pass
                #print("boss")

if __name__ == "__main__":
    from detection import Detector
    tracking = Tracking()
    #tracking.gui_video()
    tracking.gui_video()

