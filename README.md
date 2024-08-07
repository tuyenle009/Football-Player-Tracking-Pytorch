# Football Player Tracking with YOLOv5 and ResNet50

## Introduction
Welcome to my Football Player Tracking project! This repository showcases an exciting application of deep learning in sports analytics. The goal of this project is to track football players based on their team jerseys and player numbers using a combination of state-of-the-art models: YOLOv5 and ResNet50.
## Demo

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
#start to detect and classifier players
python playertracker.py
```
*''Then come to demo folder to watch results.''*

## Issues
You might encounter a few issues if you do the following:
-   Running the code without downloading the files and placing them in the corresponding folders.
-   Not installing the requirements yolov5 listed in `requirements.txt`.

## Reference
- [Yolov5](https://github.com/ultralytics/yolov5)
- [Resnet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)
- [Docker](https://hub.docker.com/)
