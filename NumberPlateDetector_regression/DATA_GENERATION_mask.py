#
# working on xml files and changing them into hdf5 file
# This data generation works only for preprocessed (detected plates by OCV)
# mask generation process
# images contain one plate or nothing
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
from imutils import perspective

#DATA_DIR = '/tmp/data'
MAIN_DIR = '/home/bayes/Academic/Research/Radarsan_01/ANPR/Finilizing_CNN/'
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
                plate[0] = elem[0][0].text
                plate[1] = elem[0][1].text
                plate[2] = elem[0][2].text
                plate[3] = elem[0][3].text
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


            # creating image masks based on recieved data ---
            img_mask = np.zeros((desired_x, desired_y))
            for plate in plates:
                img_mask[int(plate[1])+top:int(plate[3])+top, int(plate[0])+left:int(plate[2])+left] = 1


            # platel = perspective.four_point_transform(grayplt, box)
            # mas1[int(plates[0][1]) + top:int(plates[0][3]) + top,
            # int(plates[0][0]) + left:int(plates[0][2]) + left] = 255
            # plt.imshow(img_mask)

            # making data as an array - lightweight
            ImData = ImData.reshape(1,-1)
            img_mask = img_mask.reshape(1,-1)
            # post processing on the images
            im_size = ImData.shape[1]
            Sub_Data_Array = np.zeros(im_size + 2)
            Sub_Data_Array[0:im_size] = ImData
            #for row, plate in enumerate(plates):
            #    Sub_Data_Array[im_size + 4*row:im_size + 4*row + 4] = plate
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
                 gplate = h5file.create_group(h5file.root, "plate_image")
                 gmask = h5file.create_group(h5file.root, "plate_mask")
                 h5file.create_array(gplate, 'image', Sub_Data_Array, "gplate")
                 h5file.create_array(gmask, 'mask', img_mask, "gmask")
                 h5file.close()
                 print('database is created by adding image "{}"'.format(file_name))
            # # if data file exits reading the content of the pytable data file
            else:
                h5file = open_file("h5_plt_data_file_c.h5","a")
                # programming for the general case
                # The step that I should do:
                Temp_Data_file = np.vstack([h5file.root.plate_image.image[:], Sub_Data_Array])
                Temp_mask_file = np.vstack([h5file.root.plate_mask.mask[:], img_mask])
                h5file.close()
                os.remove("h5_plt_data_file_c.h5")
                h5file = open_file("h5_plt_data_file_c.h5", mode="w", title="labeled_data")
                gplate = h5file.create_group(h5file.root, "plate_image")
                gmask = h5file.create_group(h5file.root, "plate_mask")
                h5file.create_array(gplate, 'image', Temp_Data_file, "gplate")
                h5file.create_array(gmask, 'mask', Temp_mask_file, "gmask")
                h5file.close()
                print('image "{}" and its mask is successfully added to the database'.format(file_name))
                print('Data Base size is: ', Temp_Data_file.shape[0])
                print('The mask size is: ', Temp_Data_file.shape[0])
                del Sub_Data_Array, Temp_Data_file, img_mask

        except: continue

print('Data is generated')
