import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

video_path = 'C:\study_data\_data\_ydata\g20230526_154008.mp4'
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        results = model(frame)
        
        annotated_frame = results[0].plot()
        
        # 크기 조절 김대중 바봉
        cv2.namedWindow('YOLOv8 Inference', cv2.WINDOW_NORMAL)
        cv2.imshow('YOLOv8 Inference', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

# cv2.resize(annotated_frame, (200, 150))

# import cv2
# from ultralytics import YOLO

# model = YOLO('yolov8n.pt')

# video_path = 'C:/study_data/_data/_ydata/j20230526_172647.mp4'
# cap = cv2.VideoCapture(video_path)

# while cap.isOpened():
#     success, frame = cap.read()

#     if success:
#         results = model(frame)

#         annotated_frame = results.imgs[0]

#         cv2.namedWindow('YOLOv8 Inference', cv2.WINDOW_NORMAL)  # 윈도우 크기를 자유롭게 조정할 수 있는 창 생성
#         cv2.imshow('YOLOv8 Inference', annotated_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()
