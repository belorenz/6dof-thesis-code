import cv2
import yaml
import numpy as np
import utm
import os

directory = "/media/benjamin/Karla/carladlr_testdata/raw/shanghai/1/"
os.mkdir( directory + "rect", 0744 );

with open(directory + "intrinsics.yaml", "r") as file_handle:
    calibration_dict = yaml.load(file_handle)

# Undistort pixel coordinates using intrinsic calibration.
camera_matrix = np.asarray(calibration_dict["camera_matrix"]["data"]).reshape(
    3, 3)
distortion_coefficients = np.asarray(
    calibration_dict["distortion_coefficients"]["data"])

for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"): 
         # print(os.path.join(directory, filename))
	image_distorted = cv2.imread(directory + filename, cv2.IMREAD_COLOR)
	image = cv2.undistort(
    		image_distorted,
    		camera_matrix,
    		distortion_coefficients,
    		newCameraMatrix=camera_matrix)
	
	cv2.imwrite(directory + "/rect/" + filename[:-4] + "_undistorted.jpg", image)
    else:
        continue



