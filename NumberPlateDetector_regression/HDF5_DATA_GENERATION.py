#
# working on xml files and changing them into hdf5 file
# four point target creation
#
import cv2
import os
import os.path
import xml.etree.ElementTree as ET
import numpy as np
import numexpr as ne
import tables as tb
from tables import *
import matplotlib.pyplot as plt

#DATA_DIR = '/tmp/data'
MAIN_DIR = '/home/bayes/Academic/Research/Radarsan_01/ANPR/Finalizing/'
DATA_DIR = '/home/bayes/Academic/Research/Radarsan_01/labeled_plt/'
IMG_DIR = '/home/bayes/Academic/Research/Radarsan_01/ANPR/lastcd/Raw_plt_Data_3/'

#path_to_images = '/home/bayes/Academic/Research/Radarsan-01/ANPR/12_19/'

# read the xml file
for _, _, files in os.walk(DATA_DIR):
    for file in files:
        try:
            tree = ET.parse(DATA_DIR + file)
            root = tree.getroot()
            #
            file_name = root.findall('filename')[0].text
#
            objects = root.findall('object')
            #img_path =  root.findall('path')[0].text
            plates = []
            #
            for object in objects:
                elem = object.findall('bndbox')

            # checking if the object is plate,
            # in order to study the plates only:
            if object.findall('name')[0].text == 'plate' or object.findall('name')[0].text == 'pltcntnt':
                plate = np.zeros(4)
                plate[0] = eval(elem[0][0].text)
                plate[1] = eval(elem[0][1].text)
                plate[2] = eval(elem[0][2].text)
                plate[3] = eval(elem[0][3].text)
                plates.append(plate)
## ------------------------------------------
# loading corresponding images
#
            ImData = cv2.imread(IMG_DIR + file_name)
            #ImData = cv2.imread(img_path)
            try:
                ImData = cv2.cvtColor(ImData, cv2.COLOR_BGR2GRAY)
            except:
                print('This image itself is in gray scale')
#
# Preprocessing for problems resizing
#
            desired_x = 100
            desired_y = 200
            old_x = ImData.shape[0] # old_size is in (height, width) format
            old_y = ImData.shape[1]
            delta_w = desired_x - old_x
            delta_h = desired_y - old_y
            top = delta_w//2; bottom = delta_w - top
            left = delta_h//2; right = delta_h - left
            #color = [0, 0, 0] value=color
            ImData = cv2.copyMakeBorder(ImData, top, bottom, left, right, cv2.BORDER_CONSTANT)
            # checking if everything is correct
            #test_img = ImData

            ImData = ImData.reshape(1,-1)
            # post processing on the images
            im_size = ImData.shape[1]
            Sub_Data_Array = np.zeros(im_size + len(plates)*4 + 2)
            Sub_Data_Array[0:im_size] = ImData

            for row, plate in enumerate(plates):
                # padding plate
                plate[0] += left; plate[1] += top;
                plate[2] += left; plate[3] += top;

                # adding plate to the data vector
                Sub_Data_Array[im_size + 4*row:im_size + 4*row + 4] = plate

            ## checking if everything is correct
            #plate_tst = cv2.rectangle(test_img, (int(plate[0]), int(plate[1])), (int(plate[2]), int(plate[3])),
            #                         (200, 0, 0), 0)
            #plt.imshow(plate_tst)

            Sub_Data_Array[-2] = desired_x
            Sub_Data_Array[-1] = len(plates)
            #
            # The structure of sub_data_array is: Im_Matrix + all plates + number of plates
            #
            #########################################################################
            ###     Reading and writing data
            ###========================================================================
            # todo: adding the name of the files for reloading and comparison

            if not os.path.isfile('h5_plt_data_file_c.h5'):
                 h5file = open_file("h5_plt_data_file_c.h5", mode="w", title="labeled_data")
                 gplate = h5file.create_group(h5file.root, "plate_data")
                 h5file.create_array(gplate, 'datray', Sub_Data_Array, "gplate")
                 h5file.close()
                 print('database is created by adding image "{}"'.format(file_name))
            # # if data file exits reading the content of the pytable data file
            else:
                h5file = open_file("h5_plt_data_file_c.h5","a")
                # programming for the general case
                # The step that I should do:
                Temp_Data_file = np.vstack([h5file.root.plate_data.datray[:], Sub_Data_Array])
                h5file.close()
                os.remove("h5_plt_data_file_c.h5")
                h5file = open_file("h5_plt_data_file_c.h5", mode="w", title="labeled_data")
                gplate = h5file.create_group(h5file.root, "plate_data")
                h5file.create_array(gplate, 'datray', Temp_Data_file, "gplate")
                h5file.close()
                print('image "{}" is successfully added to the database'.format(file_name))
                print('Data Base size is; ', Temp_Data_file.shape[0])
                del Sub_Data_Array, Temp_Data_file

        except: continue

print('Data is generated')
