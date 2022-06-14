# Hand Rehabilitation

“Hand Rehabilitation” is a mobile application to help doctors and patients for hand diagnosis using computer vision and machine learning. It is known that if a person is severely injured in his hands, he immediately goes to the doctor, the doctor then decides to perform a surgery to overcome that injury.

Usually, if the hand is still sore and does not move well and normally, the doctor directs the patient to go and receive some physical therapy until his hands improve and return to the normal position they were in with their normal movements. The role of the physiotherapist in diagnosing the patient revolves around measuring the angles between the fingers of the patient's hand each time.The patient's diagnosis has gone through several stages.
![image](https://user-images.githubusercontent.com/105019244/173629539-8fac5de7-c85c-4e3b-8fcc-7f58d4740caf.png)


At first, the doctor used to measure these angles in a traditional way, by drawing the patient's hand on a sheet and starting to identify the lines on each finger, and then measuring the angle between them using the transferrer. The devices used began to evolve gradually, and later began using a device called the Manual Unitometer, which was used by placing it between two fingers and selecting the angle directly, but it should be noted that the disadvantages of this device, in addition to the traditional method used, are that they consume very long time and little accuracy, which requires further development in this area. Recently, doctors have been using a device called a "digital gunmeter scale" that emits two laser-like beams on two fingers and calculates the angle directly through them.

## Project Structure
```
.
+---hand3d-master
¦   +---data
¦   ¦   +---bin
¦   ¦   +---stb
¦   +---nets
¦   ¦   +---__pycache__
¦   +---utils
¦       +---__pycache__
+---Hand_Pose_Estimation_App
¦   ¦
¦   +---app
¦      -- App Files (Removed for Convenience) --
+---Models
    +---decision_tree_models
    ¦   +---decision_tree_models
    ¦       +---.ipynb_checkpoints
    +---Linear Regression
    ¦   +---.ipynb_checkpoints
    +---random_forest_models
    ¦   +---.ipynb_checkpoints
    +---svr_models
        +---svr_models
            +---.ipynb_checkpoints

```

## Technologies Used
- Machine Learning
- Computer Vision
- Android

## Tested on
- java 8
- Python 3.5.2
- tensorflow==1.3.0
- numpy==1.13.0
- scipy==0.18.1
- matplotlib==1.5.3

## Team Member
- [Ahmed Atef Azzam](https://github.com/AhmedAzzam99)
- [AbdelRahman Shata]()
- [Abdeladl Shaheen]()
- [Abdelaziz Eiwisha]()
- [Eslam Abo Alnaga]()
- [Gehad Omran]()
