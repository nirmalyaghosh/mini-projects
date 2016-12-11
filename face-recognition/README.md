# Face Recognition
Recognizes the person in a photo (*or an individual frame from a video stream*).

#### What's been done so far?
- **Train Model** : A convolutional neural network (CNN) is trained on a few hundred photos of the persons (*"subjects"*) to recognize. For the initial implementation, the CNN was trained using 10-30 second videos of our subjects - yielding approximately 250 - 750 images per subject. Uses `Keras` + `Theano`.
- **Serve Model** : A simple `Flask` app is used to predict the person in an uploaded photo.
