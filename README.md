# Football Player Tracking with YOLOv5 and ResNet50

## Introduction
Welcome to my Football Player Tracking project! This repository showcases an exciting application of deep learning in sports analytics. The goal of this project is to track football players based on their team jerseys and player numbers using a combination of state-of-the-art models: YOLOv5 and ResNet50.
## Demo
#### Overall

![output](https://github.com/user-attachments/assets/36fd3105-62ff-4398-8007-da0a8be677ff)
![Screenshot from output mp4](https://github.com/user-attachments/assets/23e33ac4-22ca-4dc2-8f7d-d53b96051f67)

#### Players Detection
![screenshot](https://github.com/user-attachments/assets/7275ab8f-b1ec-4435-a0c7-dfdcb50789fd) | ![screenshot](https://github.com/user-attachments/assets/dc96d18d-b3b8-41db-964b-db0213723fb5)|![screenshot](https://github.com/user-attachments/assets/b62e7f5a-344e-463e-9cf8-44f27e88b77d)|![screenshot](https://github.com/user-attachments/assets/3dda4a81-f3cf-460a-8880-b3012193c8a4)
|-|-|-|-|

## Project Highlights
-   **YOLOv5 for Object Detection**: Utilized to accurately detect and localize players on the field.
-   **ResNet50 for Classification**: Employed to classify the team jerseys by color and identify the player numbers.
-   **Pre-Trained Checkpoints Included**: Comes with ready-to-use checkpoints, allowing you to effortlessly apply the models for object detection and classification right away.
-   **Clear and Commented Code**: The code is easy to understand with comprehensive comments, making it accessible even for beginners.

## Installation
#### At First:
cuda 12.2 is used in this project.
```
#clone the repo
git clone https://github.com/
#go to the directory
cd Football-Player-Tracking /
```
Download the files and place them in the corresponding directory:
- [Checkpoints](https://drive.google.com/drive/folders/1f06spe35yA8MQIb15vkeYIVbKpBgdfv5?usp=drive_link)
- [Videos](https://drive.google.com/drive/folders/1ECYkrl1lAyLFkAq2HjUjtm3OkwnglN9D?usp=sharing)
 ```
  Football-Player-Tracking
  ├── checkpoints
  │   ├── player_classification.pt  
  │   ├── player_detector.pt
  ├── videos
  │   ├── football.mp4  
  └── yolov5
      ├── classify  
      ├── data
      ├── runs
      └── ...
   ...
  ```

#### Use Operating Systems:
```
#install the yolov5
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```
#### Use Docker:
```
#build docker image
docker build -t playertracker .
#run docker image
#If your laptop does not have a GPU, please drop this "--gpus 0" 
docker run -it --gpus 0 playertracker
```

## Usage
```
#start to detect and classifier players
python playertracker.py
```
The demo output can be generated with the default settings. Simply place your output.mp4 inside a demo directory 

## Issues
You might encounter a few issues if you do the following:
-   Running the code without downloading the files and placing them in the corresponding folders.
-   Not installing the requirements yolov5 listed in `requirements.txt`.

## Reference
- [Yolov5](https://github.com/ultralytics/yolov5)
- [Resnet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)
- [Docker](https://hub.docker.com/)
