# Face Recognition
Recognizes the person in a photo (*or an individual frame from a video stream*).

#### What's been done so far?
- **Train Model** : A convolutional neural network (CNN) is trained on a few hundred photos of the persons (*"subjects"*) to recognize. For the initial implementation, the CNN was trained using 10-30 second videos of our subjects - yielding approximately 250 - 750 images per subject. Uses `Keras` + `Theano`.
- **Serve Model** : A simple `Flask` app is used to predict the person in an uploaded photo.

#### Getting Started
I assume you already have VirtualBox (version 5+) installed, if you don't, please [download](https://www.virtualbox.org/wiki/Downloads) and install it. You will also need to [download and install Vagrant](http://www.vagrantup.com/downloads.html) if you haven't previously done so.

1. Change into the `face-recognition` directory and run `vagrant up`. This create the 2 VMs required for this mini-project.
2. Once you are done for the day, `vagrant suspend`. To resume, `vagrant resume`.
3. To destroy the VMs, `vagrant destroy`.
