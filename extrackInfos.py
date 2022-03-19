""""
Provides the number of classes from .txt files
"""

import cv2
import random

class Info:
    def __init__(self):
        self.cntt = 0
        self.testFilePath = "test.txt"    #"data/test.txt"
        self.trainFilePath = "train.txt"  #"data/train.txt"
        self.testFile = open(self.testFilePath, 'r')
        self.trainFile = open(self.trainFilePath, 'r')
        self.classesList = [[0],[0],[0],[0],[0],[0]] 
        self.label_list = {"0": "car",
                        "1": "truck",
                        "2": "van",
                        "3": "big-truck",
                        "4": "motorbike",
                        "5": "pedestrian"}

    def get_file_path(self):
        while self.trainFile:
            self.lineTest =self.testFile.readline()
            self.lineTrain =self.trainFile.readline()
            name = 'data'
            if self.lineTrain[:len(name)] == name and self.lineTrain != '': # and self.cntt != 4: 
                #print(self.lineTrain)
                self.get_file(self.lineTrain,self.lineTest)

            elif self.lineTrain == '':
                print("DONE!!")
                break

    def get_file(self, trainPath, testPath):
        self.imgTrain = cv2.imread(trainPath[:-1])
        #self.imgTest = cv2.imread(testPath[:-1])

        self.trainTxt = trainPath[:-5]
        #self.testTxt = testPath[:-5]

        trainF = open(self.trainTxt + ".txt")
        #testF = open(self.testTxt + ".txt")
        while trainF:
            self.train = trainF.readline()
            if self.train != '':
                self.classesList[int(self.train[0])][0]+=1 
                #print(self.classesList)
                #
                xmin, ymin, xmax, ymax = self.imgInfo()
                self.imgTrain = cv2.rectangle(self.imgTrain, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
                cv2.putText(self.imgTrain, self.label_list[self.train[0]], (xmin,ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
                #cv2.imshow("train", cv2.resize(self.imgTrain, (608,608)))

                # if ((xmax-xmin)*(ymax-ymin))<5000:
                #     cv2.imshow("single img", self.imgTrain[ymin:ymax,xmin:xmax])
                #     print("Alan: ", (xmax-xmin)*(ymax-ymin))
                #     cv2.waitKey(10)

                # cv2.imshow("test", self.imgTest)
                # if cv2.waitKey(10) & 0xFF == ord("q"):
                #     cv2.imwrite("/home/zfg/Desktop/saved/fethiye/"+ str(random.random()) + ".jpg", self.imgTrain)
                #     print("saved!!!!!!")
                # #
            else:
                break
                # self.cntt += 1
                # if self.cntt == 4:
                #     break
                # cv2.imwrite("/home/zfg/Desktop/saved/2/"+ str(random.random()) + ".jpg", self.imgTrain)
                # break

    def imgInfo(self):
        h, w, d = self.imgTrain.shape
        bbox_width = float(self.train[20:28])*w
        bbox_height = float(self.train[29:37])*h
        center_x = float(self.train[2:10])*w        
        center_y = float(self.train[11:20])*h 
        xmin = int(center_x - (bbox_width / 2))
        ymin = int(center_y - (bbox_height / 2))
        xmax = int(center_x + (bbox_width / 2))
        ymax = int(center_y + (bbox_height / 2))
        return xmin, ymin, xmax, ymax

    def show_single_pic(self):
        pass

    def gui(self):
        self.get_file_path()
        print(self.classesList)

if "__main__" == __name__:
    IF = Info()
    IF.gui()