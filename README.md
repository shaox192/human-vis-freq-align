# Does aligning Neural Network Frequency Bands to Human Vision improve robustness?

This is the official code repo for CS543 CV FA24 course project

***
### Folders and Files

**delta_scripts/**: 
Job submission scripts on delta (GPU).

**attack_alg/**
attack algorithms

**models**
- bandpass layer
- blur layer

**train.py**
This is the main script for training, to use this, first make sure that the **train** section in ["run.sh"](./run.sh) is commented out, and you have the data files with appropriate paths, and then run it with the following command:
```sh
bash run.sh
```
I have the toy dataset on the [gdrive](https://drive.google.com/drive/folders/12V9IFrhZiCWYsPAO7M5bDSMBCWcg4u4G?usp=sharing). This has 6 categories, each with 6 images, but you can subsample the categories in the toy.txt. You can run it for debugging on your local machine.

Trained model weights are also avaibale on the [gdrive](https://drive.google.com/drive/folders/13a3pmUB7ucv4j1ZeKF5uWbxSw-Wtivn9?usp=sharing)

**attack.py**
This is the main script for attacking the model, to use this, first make sure that the **attack** section in ["run.sh"](./run.sh) is commented out, and you have the data files with appropriate paths, and then run it with the following command:
```sh
bash run.sh
```
For debugging, you can use the None model, so vanilla resnet classifier, thus without needing the model weights.


### Todo

- [x] **Implement model with blur layer**:  
    skeleton is provided in [blur_net.py](./models/blur_net.py). I have the TODOs. The blur layer is taken from [this](https://github.com/lizhe07/blur-net/blob/master/blurnet/models.py). Then need to update the [train.py](./train.py) to enable --append-layer "blur", I also have a TODO in there.

- [x] **Add natural attack**:  
    I have a skeleton in [natural.py](./attack_alg/natural.py), we basically want to migrate [this](https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py) here. They have many types, we can just use the main 15 in their [picture](https://github.com/hendrycks/robustness/tree/master) for now. The usage of this function is in [attack.py](attack.py), which is now written for adv attacks only. The goal is to make minimal changes to this file to enable passing in natural attacks. **This is lower priority than the other two, but we will need this especially once we see results from the blur layer model above**.

- [x] **Double-check bandpass**:  
    Double check the current bandpass layer setup for bugs. Then experimenting with: starting from a very large sigma, gradually narrow down the frequency band to see the effect on the robustness performance. This is both for making sure that the code works and see if it is the overly narrow band that's causing the low perf.

