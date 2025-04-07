from PyQt5.QtWidgets import QFileDialog, QMainWindow,  QGraphicsScene,  QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from result_fasterrcnn import Result  # 導入子窗口
from result_yolov8 import Result_Yolov8
import ui as UI
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
import torchvision.models as models
from torchsummary import summary
import torch.nn.functional as F


class MainWindow_controller(QMainWindow):
    def __init__(self):
        super(MainWindow_controller, self).__init__() # in python3, super(Class, self).xxx = super().xxx   # 所以可簡寫成super().__init__()
        self.ui = UI.Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton.clicked.connect(self.openDirectoryDialog)
        self.ui.pushButton_2.clicked.connect(self.pre_image)
        self.ui.pushButton_3.clicked.connect(self.next_image)
        self.ui.pushButton_4.clicked.connect(self.detection)
        self.total = 0
        self.correct = 0
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.ious = []
        # self.ui.text_edit.textChanged.connect(self.input_text)    
        

    
    def openDirectoryDialog(self):          # 選擇目錄
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder')    # QFileDialog.getExistingDirectory 是一個class method，不需要實例化 QFileDialog 物件就可以直接使用，這種方法通常用於調用靜態對話框。
        if folder:
            self.i = 0 
            self.image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            print(self.image_files[2])
            print(folder)  
            self.folder = folder  
            self.change_label_text()

     
    def pre_image(self):        
        self.i -= 1
        if self.i < 0:
            self.i = 0
        self.change_label_text()        
        
    def next_image(self): 
        self.i += 1   
        if self.i >= len(self.image_files): 
            self.i = len(self.image_files) - 1  
        self.change_label_text()  
            
    def change_label_text(self):
        self.ui.label.setText(self.image_files[self.i])
        self.filename = self.image_files[self.i]
        
    def detection(self):
        result  = Result(self.folder, self.filename, self.total, self.correct, self.true_positive, self.false_positive, self.false_negative, self.ious)
        # result  = Result_Yolov8(self.folder, self.filename, self.total, self.correct, self.true_positive, self.false_positive, self.false_negative, self.ious)
        print(result.correct)
        print(result.total)
        self.total = result.total
        self.correct = result.correct
        self.true_positive = result.true_positive
        self.false_positive = result.false_positive
        self.false_negative = result.false_negative        
        self.ious = result.ious
        # 檢查數組是否為空
        if len(result.ious) > 0:
            self.iou = np.mean(result.ious)
        else:
            self.iou = None 
        self.accuracy = result.correct / result.total if result.total > 0 else 0
        self.presicion = result.true_positive / (result.true_positive + result.false_positive) if (result.true_positive + result.false_positive) > 0 else 0
        self.recall = result.true_positive / (result.true_positive + result.false_negative) if (result.true_positive + result.false_negative) > 0 else 0
        self.change_bottom_label_text()
        
    def change_bottom_label_text(self):
        new_iou = "IoU : " + str(self.iou)
        self.ui.label_2.setText(new_iou)
        new_accuracy = "Accuracy : " + str(self.accuracy)
        self.ui.label_3.setText(new_accuracy)
        new_presicion = "Presicion : " + str(self.presicion)
        self.ui.label_4.setText(new_presicion)
        new_recall = "Recall : " + str(self.recall)
        self.ui.label_5.setText(new_recall)
    
    
    