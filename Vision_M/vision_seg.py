import os
import cv2
import numpy as np
from ultralytics import YOLO

# === STEP 1: TRAINING YOLOv8 SEGMENTATION MODEL ===
class VisionSegmentation():
    def __init__(self, model_path):
        self.model_path = model_path

    def train_model(self):
        # Replace with the path to your data.yaml file
        data_yaml = "./data.yaml"

        # Load a YOLOv8 segmentation model (you can also try yolov8s-seg.yaml or yolov8m-seg.yaml)
        model = YOLO("yolo11m-seg.pt")

        # Train the model
        model.train(
            data=data_yaml,
            epochs=200,
            imgsz=512,
            batch=16,
            project="runs_seg/train",  
            name="rainbowflow-seg",
            device="cuda",
            patience=50  # Early stopping patience
        )
        model.val(data="./data.yaml", split='test')


    # === STEP 2: INFERENCE AND CENTROID EXTRACTION ===

    def run_inference_and_extract_centroids(self, image_path):
        # Load the trained model
        model = YOLO(self.model_path)

        # Run inference
        results = model(image_path)[0]

        # Show image with masks
        results.show()

        if not results.masks:
            print("No masks found.")
            return

        print("\n--- Centroid positions ---")
        for i, mask_tensor in enumerate(results.masks.data):
            # Convert tensor to NumPy boolean mask
            mask = mask_tensor.cpu().numpy().astype(np.uint8)

            # Find contours in the binary mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    print(f"Object {i+1}: Centroid at (x={cx}, y={cy})")
                else:
                    print(f"Object {i+1}: Cannot compute centroid (zero area)")


    def run_live_inference_with_centroids(self, cam_index=0):
        # Cargar modelo
        model = YOLO(self.model_path)

        # Abrir cámara
        cap = cv2.VideoCapture(cam_index)
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

        if not cap.isOpened():
            print("Error. No Camera Found")
            return

        print("Press 'q' to exit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Ejecutar inferencia
            results = model(frame)[0]

            # Dibujar máscaras sobre el frame
            annotated_frame = results.plot()

            # Extraer y mostrar centroides si hay máscaras
            if results.masks:
                for i, mask_tensor in enumerate(results.masks.data):
                    mask = mask_tensor.cpu().numpy().astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            position = (cx, cy)
                            '''
                            depth = depth_frame.get_distance(cx, cy)
                            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                            x, y, z = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)
                            position = np.array([x, y, z])  # In meters
                            '''
                            cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)
                            cv2.putText(annotated_frame, f"({cx},{cy})", (cx + 5, cy - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Mostrar el frame
            cv2.imshow("YOLOv8 Live Segmentation", annotated_frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return position

# === MAIN ===
if __name__ == "__main__":
    # STEP 1: Uncomment to train the model
    vision = VisionSegmentation(model_path="./runs_seg/train/rainbowflow-seg13/weights/best.pt")
    vision.train_model()

    # STEP 2: Provide the image path and the trained model path
    image_path = "./train/images/20250613_130004_png.rf.0a0c9364f883baeacdc6e8f8efae1296.jpg"
    model_path = "./runs_seg/train/rainbowflow-seg13/weights/best.pt" #Number 13 best

    vision.run_inference_and_extract_centroids(image_path)
    #positon = vision.run_live_inference_with_centroids()