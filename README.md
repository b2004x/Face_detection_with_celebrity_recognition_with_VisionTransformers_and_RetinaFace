Face Detection and Celebrity Classification using Vision Transformer and RetinaFace
Project Overview

This project is divided into two main parts:

1. Face Detection and Face Recognition

The system detects human faces in an image, crops them, saves the detected face, and compares it with another image to check if they belong to the same person.
It uses deep-learning face embeddings and similarity scoring for fast and accurate recognition.

2. Celebrity Classification

The model detects faces in an image and predicts which celebrity the face most closely resembles.

Features

Face Detection: Detects multiple faces, draws bounding boxes, and saves coordinates using RetinaFace.

Face Recognition: Detects faces and matches them with stored face embeddings using ViT or MTCNN.

Celebrity Classification: Predicts the celebrity look-alike using Vision Transformer.

Samples
Face Detection

Face Recognition




Celebrity Classification

Dataset
Face Detection & Face Recognition

The RetinaFace model uses the WiderFace Dataset.

Dataset: https://drive.google.com/file/d/11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS/view

data/
└── widerface/
    ├── train/
    │   ├── images/
    │   └── label.txt
    └── val/
        ├── images/
        └── wider_val.txt


wider_val.txt contains only validation file names (no label information).

Celebrity Classification

Vision Transformer uses the Celebrity Face Image Dataset.
Dataset: https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset

data/
└── celebrity_faces/
    ├── Angelina_Jolie/
    │   ├── 001_fe3347c0.jpg
    │   └── ...
    ├── Brad_Pitt/
    │   └── ...
    ├── Denzel_Washington/
    │   └── ...
    ├── Hugh_Jackman/
    │   └── ...
    ├── Jennifer_Lawrence/
    │   └── ...
    ├── Johnny_Depp/
    │   └── ...
    ├── Kate_Winslet/
    │   └── ...
    ├── Leonardo_DiCaprio/
    │   └── ...
    ├── Megan_Fox/
    │   └── ...
    ├── Natalie_Portman/
    │   └── ...
    ├── Nicole_Kidman/
    │   └── ...
    ├── Robert_Downey_Jr/
    │   └── ...
    ├── Sandra_Bullock/
    │   └── ...
    ├── Scarlett_Johansson/
    │   └── ...
    ├── Tom_Cruise/
    │   └── ...
    ├── Tom_Hanks/
    │   └── ...
    └── Will_Smith/
        └── ...

Tech Stack
Backend / API

FastAPI

Pydantic

Machine Learning / Deep Learning

PyTorch

Transformers (HuggingFace ViT)

facenet-pytorch (InceptionResnetV1, MTCNN)

scikit-learn

RetinaFace: https://github.com/yakhyo/retinaface-pytorch

Vision Transformer: https://arxiv.org/pdf/2010.11929

MTCNN: https://github.com/ipazc/mtcnn

Image Processing

Pillow (PIL)

torchvision

Data Handling

NumPy

joblib

JSON

base64

Frontend / Interface

Streamlit

Utilities

tempfile, io, os, pathlib

subprocess

Installation
git clone <your_repo_url>
cd FACE_DETECTION - COPY

Model Performance
Face Detection
Epoch: 100/100 | Batch: 400/400
Loss Localization: 0.5011
Classification: 0.8221
Landmarks: 0.7289
LR: 0.00001000
Average batch loss: 2.5618900

Celebrity Classification

Classification Report:

accuracy: 0.8500
macro avg F1-score: 0.8291
weighted avg F1-score: 0.8488


Confusion Matrix:


How to Run
1. Start Backend
cd demo/back-end
uvicorn api:app --reload

2. Start Frontend

Open another terminal:

cd demo/front-end
streamlit run app.py

Front-end Screenshots

Choose an image to detect face & save embedding:


Match detected face with database:


Predict celebrity:


References

https://github.com/yakhyo/retinaface-pytorch

https://arxiv.org/pdf/2010.11929

https://github.com/ipazc/mtcnn
