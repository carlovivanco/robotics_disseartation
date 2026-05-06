'''This code intends to identify different types 
of bolts'''
import torch
import pyrealsense2 as rs
import timm
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from ultralytics import settings
import cv2
import numpy as np


class Vision():
    # Load a YOLOv5n model pretrained on COCO
    def __init__(self):
        self.CLASS_NAMES = ['10.9', '12.9', '4.6', '5.6', '5.8', '6.8', '8.8', '9.8']

    def train_model(self):
        #settings.update({"runs_dir": "C:/Users/Carlo/Desktop/Essex/Thesis/code/Vision/runs_detection", "tensorboard": True})
        model = YOLO("yolo11l.pt")

        # Display model info (optional)
        model.info()

        # Train the model with your custom dataset
        results = model.train(
            data="C:/Users/Carlo/Desktop/Essex/Thesis/code/Vision/data.yaml",  # path to your .yaml file
            epochs=50,
            imgsz=640,
            batch=8,  # you can adjust batch size depending on your GPU
            project="runs_detection/train",  # where results will be saved
            name="yolo11m",# name of the experiment
            device = 'cuda'
        )
        return results

    def preprocess_for_ocr(self,roi):
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Denoise while preserving edges
        #denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Equalize histogram to enhance contrast
        #eq = cv2.equalizeHist(denoised)

        # Apply adaptive thresholding suitable for printed text
        thresh = cv2.adaptiveThreshold(gray, 255, 
                                    cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY_INV, 
                                    31, 15)

        return thresh
    # Cargar clases manualmente si no se usa train_dataset.classes

    # Modelo de clasificación (ViT)
    def load_classifier(self,model_path, num_classes=8):
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        model.eval()
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        return model

    # Transforms para ViT
    def preprocess_for_classification(self,roi):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        return transform(img_pil).unsqueeze(0)

    def euler_to_quaternion(self, roll_deg, pitch_deg, yaw_deg):
        roll = np.radians(roll_deg)
        pitch = np.radians(pitch_deg)
        yaw = np.radians(yaw_deg)

        c1 = np.cos(yaw / 2)
        s1 = np.sin(yaw / 2)
        c2 = np.cos(pitch / 2)
        s2 = np.sin(pitch / 2)
        c3 = np.cos(roll / 2)
        s3 = np.sin(roll / 2)

        x = s3 * c2 * c1 - c3 * s2 * s1
        y = c3 * s2 * c1 + s3 * c2 * s1
        z = c3 * c2 * s1 - s3 * s2 * c1
        w = c3 * c2 * c1 + s3 * s2 * s1

        return np.array([x, y, z, w])

    def test_with_webcam(self, model_path="./runs_detection/train/yolov11m2/weights/best.pt", 
                        classifier_path="Vision2/runs/train/vit_classification.pth", 
                        cam_index=0, ready=1):
        
        detector = YOLO(model_path)
        classifier = self.load_classifier(classifier_path, num_classes=len(self.CLASS_NAMES))

        cap = cv2.VideoCapture(cam_index)
        #For RealSense Camera
        '''pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        align = rs.align(rs.stream.color)'''

        if not cap.isOpened():
            print("Cannot open webcam")
            return None, None, ready

        position = None
        orientation_quat = None

        while ready:
            ret, frame = cap.read()
            #For RealSense Camera
            '''
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            '''
            if not ret:
                print("Failed to grab frame")
                break

            results = detector.predict(source=frame, show=False, conf=0.6)
            boxes = results[0].boxes

            annotated_frame = frame.copy()

            if len(boxes) == 0:
                cv2.imshow("Bolt Detection + Classification", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    ready = 0
                continue  # no detections

            # Only consider first detected bolt (you can modify this to loop over all)
            box = boxes[0]
            cls_id = int(box.cls.cpu())
            xyxy = box.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, xyxy)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cz = 0  # Assuming Z=0
            position = np.array((cx, cy, cz))
            #For RealSense Camera
            '''
            depth = depth_frame.get_distance(cx, cy)
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            x, y, z = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)
            position = np.array([x, y, z])  # In meters
            '''

            # Draw center and bounding box
            cv2.circle(annotated_frame, (cx, cy), radius=4, color=(0, 0, 255), thickness=-1)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            bolt_roi = frame[y1:y2, x1:x2]
            if bolt_roi.size == 0:
                cv2.imshow("Bolt Detection + Classification", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    ready = 0
                continue

            roi_gray = cv2.cvtColor(bolt_roi, cv2.COLOR_BGR2GRAY)
            roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
            edges = cv2.Canny(roi_blur, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            orientation_deg = 0  # Default if not detected
            for cnt in contours:
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) == 6:
                    # Draw hexagon edges on annotated_frame
                    for i in range(6):
                        pt1 = tuple(approx[i][0] + np.array([x1, y1]))
                        pt2 = tuple(approx[(i + 1) % 6][0] + np.array([x1, y1]))
                        cv2.line(annotated_frame, pt1, pt2, (0, 255, 0), 2)
                    pt_a = tuple(approx[0][0] + np.array([x1, y1]))
                    pt_b = tuple(approx[3][0] + np.array([x1, y1]))
                    cv2.line(annotated_frame, pt_a, pt_b, (255, 0, 255), 2)

                    dx = pt_b[0] - pt_a[0]
                    dy = pt_b[1] - pt_a[1]
                    orientation_deg = np.degrees(np.arctan2(dy, dx))
                    break

            # Convert 2D orientation to quaternion (roll=0, pitch=0, yaw=orientation_deg)
            orientation_quat = self.euler_to_quaternion(0, 0, orientation_deg)

            # Classification
            classifier_device = next(classifier.parameters()).device
            input_tensor = self.preprocess_for_classification(bolt_roi).to(classifier_device)
            with torch.no_grad():
                output = classifier(input_tensor)
                pred_class = output.argmax(dim=1).item()
                confidence = torch.nn.functional.softmax(output, dim=1)[0, pred_class].item()

            label_text = f"Bolt {self.CLASS_NAMES[pred_class]} ({confidence:.2f})"
            print(label_text)
            print(f"Position: {position}, Orientation (deg): {orientation_deg:.2f}")
            print(f"Orientation (quat): {orientation_quat}")

            # Put label text on the annotated frame above the box
            cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Show annotated frame with all drawings
            cv2.imshow("Bolt Detection + Classification", annotated_frame)
            # Exit on space key
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        cap.release()
        cv2.destroyAllWindows()

        return position, orientation_quat, ready


if __name__ == "__main__":
    # Uncomment the next line if you want to train first
    #train_model()
    #torch.cuda.empty_cache()
    #print(torch.cuda.memory_summary())
    vision_m = Vision()
    #vision_m.train_model()

    # Test with webcam
    #test_with_image("testimg2.png")
    vision_m.test_with_webcam()
