## AI enhanced Intelligent Systems and Robotics for Flexible Manufacturing Systems
This project aims to implement a DRL algorithms to automate the disassembly stage for the industrial robots. In this case for unscrew-driving an M5 Bolt.

# How to Run the DRL Code

1. **Open Anaconda Prompt or PowerShell.**

2. **Activate your base environment:**
Ex: C:/Users/Carlo/miniconda3/Scripts/activate

3. **Activate the project environment:**
```sh
conda activate mujoco_franka
```

4. **Install all the requirements:**
```sh
pip install -r requirements1.txt
```

5. **Run any Python script (example for Imitation Learning):**
```sh
python Imitation.py
```

or 
```sh
python DRL_BoltSD.py
```

**Note:**  
- Make sure you are in the correct folder and all dependencies are installed.
- For ROS2 integration or robot control, ensure ROS2 is sourced and running.
- You can uncomment the "training" functions in both DRL and Imitation codes to see the perfomance of them during training, but the codes are ready to run for Test Only to see the actual results of the project.

---

## Vision Segmentation Instructions

This section explains how to train, test, and run live inference using the YOLOv8 segmentation model for bolt and nut detection.

### 1. Install Requirements

Make sure you are in the `Vision_M` folder:

```sh
cd Vision_M
pip install -r requirements.txt
python vision_seg.py
```
**Note:**  
- Make sure you are in the correct folder and all dependencies are installed.
- To run the static image test Make sure to set the correct image_path and model_path in the __main__ section of vision_seg.py.
- To run live segmentation and centroid extraction using your webcam, uncomment the line in vision_seg.py: # position = vision.run_live_inference_with_centroids().
- For RealSense camera integration, see the commented code in vision_seg.py.
- You can change the YOLO model type (e.g., yolov8s-seg.pt, yolov8m-seg.pt) in the script.
- Make sure your CUDA drivers are installed if you want to use GPU acceleration.

