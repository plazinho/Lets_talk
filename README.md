## Let's talk - ASL (American Sign Language) recognition
### This project was completed in 10 days by:
- https://github.com/plazinho
- https://github.com/IlyaGaluzinskiy
- https://github.com/aabdysheva

### Let's talk was done in order to recognize and translate ASL gestures in real-time using computer vision
- Mediapipe model (https://google.github.io/mediapipe/) was used to make precise keypoint localization of 21 3D hand-knuckle coordinates
 
![hand_landmarks-new](https://user-images.githubusercontent.com/88561819/139060508-fc1e68a4-cfc3-4406-8bde-a6f126074f7f.jpg)

- Let's talk currently able to recognize 39 gestures: 26 alphabet letters, 10 words and 3 special gestures. 12 of them have movement
- LSTM model for sign recognition was build using Keras and trained on dataset, that was collected during development (currently includes more than 5000 samples)
- user guide and list of available signs are located in the UI
- required libraries can be installed using requirements.txt file

Folder "for_model_training" includes scripts, that will allow you to create your own dataset.
