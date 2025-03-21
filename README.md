Unusual-Sus-behavior-in-ATM-machines

**Overview**
This project is designed to detect unusual activities inside ATMs using deep learning and computer vision. The system utilizes the YOLOv8 object detection model to monitor ATM surveillance videos and detect suspicious behavior such as unauthorized access, weapon detection, and attempts to block security cameras.

**Dataset**
        > We used the ATMA-V Dataset, which contains surveillance footage of ATM activities. 
        > The dataset includes various video files featuring normal and suspicious activities.

**Tools & Technologies Used**
        > Programming Language: Python
        > Deep Learning Framework: YOLOv8 (Ultralytics)
        > Computer Vision Library: OpenCV
        > Libraries Used:
                   kagglehub
                   cv2
                   torch
                   numpy
                   ultralytics

**Project Setup**
    **Prerequisites**
        > Ensure you have Python (>=3.8) installed. You can install the required dependencies using the following command:
                   pip install kagglehub opencv-python torch numpy ultralytic
 
   **Running the Project**
        1) Clone the repository:
                   git clone https://github.com/yourusername/unusual-behaviour-atm.git
                   cd unusual-behaviour-atm
        2) Place the ATMA-V Dataset videos inside the dataset/ATMA-V Dataset/videos folder.
        3) Run the detection script:
                   python atm_security.py
        4) The system will process each video in the dataset and highlight unusual activities.
        5) If any suspicious activity is detected, a warning (DANGER) will be displayed, and the footage will be recorded.
        6) Press q to quit execution.


**Execution Flow**
        1) Load the YOLOv8 model (yolov8n.pt).
        2) Iterate through ATM surveillance videos and apply object detection.
        3) Identify classes of interest: person, hand, weapon, card, phone, wallet.
        4) Detect suspicious activities such as:
                   > Touching non-screen areas
                   > Bringing a weapon into the ATM
                   > Blocking the camera
        5) If unusual activity is detected:
                   > The system raises an alert. 
                   > The video footage is recorded for 10 seconds.
        6) The processed video frames are displayed with bounding boxes and labels.


**Future Enhancements**
         > Integration with real-time ATM CCTV feeds.
         > AI-based anomaly detection for improved accuracy.
         > SMS/email notifications for security alerts.
  
