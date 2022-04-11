# Optical Character Recognition of Kannada digits

## Description

This project involves a program that identifies a Kannada digit that is given as input by making a custom dataset to run a lazy trained KNN Machine Learning Algorithm. Another python program is run to convert the input image dataset into MNIST format which is then used for the machine learning process. A paper written for this project can be found [here](https://drive.google.com/file/d/1VTlCOB1Yy2m10Lx1uPJEXPA8VQNRIjUL/view?usp=sharing).

## Setup
- Download or Clone the project and run `ocr.py`

- If any error is encountered, check if the packages `numpy` and `imagio` have been imported properly in `ocr.py` and `new_data/convert.py` .

## Usage 
- Run `ocr.py` to run the project.

- The program parses the images in `our_input` and predicts labels for them. These labels are then compared with the actual labels (known from the file name) of the images to determine the accuracy of the program. 

![image](https://user-images.githubusercontent.com/73631606/162838803-3c7ffbd3-856a-422a-9c44-32f581b0d821.png)

- The project only works on input images that are black and white having a dimension of 28 x 28 pixels. 
