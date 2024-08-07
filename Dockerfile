FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /playertracker

COPY checkpoints ./checkpoints
COPY demo ./demo
COPY images ./images
COPY src ./src
COPY videos ./videos
COPY playertracker.py ./playertracker.py

RUN apt-get update
RUN apt-get install vim ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python
RUN apt install git -y
RUN git clone https://github.com/ultralytics/yolov5
RUN pip install -r yolov5/requirements.txt