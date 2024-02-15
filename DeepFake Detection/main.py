import numpy as np
import cv2
import torch
import torchvision.models as models
import xgboost as xgb

def detect_faces(video_frames):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_images = []
    for frame in video_frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (128, 128))
            face_images.append(face_resized)
    return face_images

def extract_features(face_images):
    #InceptionResNetV2 model
    base_model = models.inception_v3(pretrained=True, aux_logits=True)
    model = torch.nn.Sequential(*list(base_model.children())[:-1])
    model.eval()

    resized_images = [cv2.resize(img, (128, 128)) for img in face_images]

    preprocessed_images = torch.tensor(resized_images).float() / 255.0
    preprocessed_images = torch.nn.functional.interpolate(preprocessed_images, size=(128, 128))

    with torch.no_grad():
        features = model(preprocessed_images)
    
    return features.squeeze()

def train_xgboost(features, labels):
    clf = xgb.XGBClassifier()
    clf.fit(features, labels)
    return clf

def detect_deepfakes(video_frames):
    face_images = detect_faces(video_frames)
    features = extract_features(face_images)
    clf = train_xgboost(features, labels)
    probabilities = clf.predict_proba(features)
    return probabilities

def load_video_frames(video_path):
    video_capture = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
    video_capture.release()
    return frames

video_path = "D:/My Folders/Club Gamma Session/Generative AI from Scratch/Videos and Pictures/Raw Videos/WhatsApp Video 2024-02-10 at 20.14.41_709963a6.mp4"
video_frames = load_video_frames(video_path)
probabilities = detect_deepfakes(video_frames)