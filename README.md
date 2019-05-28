# Bachelor Thesis

This thesis is a comparison between three different training algorithms when training a Convolutional Neural Network for classifying road signs. The algorithms that were compared were Gradient Descent, Adadelta, and Adam. For this study the German Traffic Sign Recognition Benchmark (GTSRB) was used, which is a scientifically relevant dataset containing around 50000 annotated images. A combination of supervised and offline learning was used and the top accuracy of each algorithm was registered. Adam achieved the highest accuracy, followed by Adadelta and then Gradient Descent. Improvements to the neural network were implemented in form of more convolutional layers and more feature recognizing filters. This improved the accuracy of the CNN trained with Adam by 0.76 percentage points.

In this repository, each version of the code with a higher version number is an improvement to the previous version with regards to the accuracy of the network.
