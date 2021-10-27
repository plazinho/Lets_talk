## Let's talk - ASL (American Sign Language) recognition
### This project was completed in 10 days by:
- https://github.com/plazinho
- https://github.com/IlyaGaluzinskiy
- https://github.com/aabdysheva

### Let's talk was created in order to recognize and translate ASL gestures in real-time using computer vision
- Let's talk currently able to recognize 39 gestures: 26 alphabet letters, 10 words and 3 special gestures. 12 of them are gestures that have movement
- Let's talk was created using MediaPipe, OpenCV, Keras, PyQt
- LSTM model for sign recognition was built using Keras and trained on dataset, that was collected during development (currently includes more than 5000 samples)
- Prediction model architecture: 3 LSTM layers with ReLU activation and output fully-connected layer with softmax activation
- User guide and list of available signs are located in the UI

![Kirill](https://user-images.githubusercontent.com/88561819/139092550-d1b2ef50-641f-467b-a74d-7a6550432974.gif)

### Interface:
- Top left corner of the screen - predictions
- Field at the top of the screen - detected gestures


### The process of sign recognition consists of following steps:
- Video from camera is received with a help of OpenCV
- Video frames are passed to MediaPipe Holistic model that detects hands, adds landmarks (21 landmarks per hand) and records their coordinates.
- Landmarks coordinates are extracted, organized and passed to the LSTM model for prediction.
- The visualization of model prediction is implemented in the top left corner of the program screen (see gif above). 
- Detected signs are translated to english and displayed on top of the screen (see gif above).

### In order to start "Let's Talk" on your machine:
- Clone repository
- Install required libraries (pip install -r requirements.txt)
- Run 'main_ui.py'

### Create your own dataset:
Folder "for_model_training" includes scripts that were used for creation of dataset.
- script 'for_checking_your_camera_and_mediapipe_model' - to see how MediaPipe model actually works with a vizualization of hand's landmarks

#### Steps for dataset creation:
- script 'creating_folders_for_dataset' - for creation of signs data folders
- script 'creating_dataset' - to record coordinates for a certain sign and place it in previously created folder (dataset for the program was built by recording coordinates, specific for each gesture. To widen the range of the dataset it was recorded by 3 people).
- script 'preparing_data' - to organize and label data in order to pass it to the model
- script 'train_LSTM_model' - includes necessary functions to process the data and pass it to the model
