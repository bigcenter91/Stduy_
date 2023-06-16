import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# COCO 클래스 이름과 색상 설정
class_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
    'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
class_colors = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                [255, 0, 255], [0, 255, 255]]

# 모델 빌드
def build_model():
    model = tf.keras.applications.MaskRCNN(
        backbone=tf.keras.applications.ResNet50V2(
            weights='imagenet', include_top=False),
        num_classes=len(class_names))
    return model

# 이미지 읽기 및 전처리
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return image

# 결과 시각화
def visualize_results(image, boxes, masks, class_ids):
    for i in range(len(boxes)):
        color = class_colors[class_ids[i]]
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)
        image = draw_bbox(image, boxes[i], color)

    plt.imshow(image)
    plt.axis('off')
    plt.show()

# 마스크 적용
def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c],
            image[:, :, c]
        )
    return image

# 바운딩 박스 그리기
def draw_bbox(image, box, color):
    image = cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), color, 2)
    return image

# 이미지에서 물체 감지 및 분할
def detect_and_segment(image_path):
    image = preprocess_image(image_path)
    model = build_model()
    model.load_weights('mask_rcnn_weights.h5')

    # 이미지 추론
    results = model.detect([image])[0]
    masks = results['masks']
    boxes = results['rois']
    class_ids = results['class_ids']

    # 결과 시각화
    visualize_results(image, boxes, masks, class_ids)

# 이미지 경로
image_path = 'C:\Stduy_\con_prac\test\WIN_20230607_19_24_51_Pro.jpg'


# 이미지에서 물체 감지 및 분할 수행
detect_and_segment(image_path)
