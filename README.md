# ssd_tensorflow
Single Shot Multibox Detector (SSD) ([paper](https://arxiv.org/abs/1512.02325)) implementation in TensorFlow, in development.

Results of some hand-picked test images through an experimental run with MS COCO, some good and some bad:

<img src="https://raw.githubusercontent.com/seann999/ssd_tensorflow/master/images/Screenshot%20from%202016-11-02%2004%3A28%3A44.png" width="300"/>
<img src="https://raw.githubusercontent.com/seann999/ssd_tensorflow/master/images/Screenshot%20from%202016-11-02%2004%3A29%3A38.png" width="300"/>
<img src="https://raw.githubusercontent.com/seann999/ssd_tensorflow/master/images/Screenshot%20from%202016-11-02%2004%3A31%3A01.png" width="300"/>
<img src="https://raw.githubusercontent.com/seann999/ssd_tensorflow/master/images/Screenshot%20from%202016-11-02%2004%3A31%3A58.png" width="300"/>
<img src="https://raw.githubusercontent.com/seann999/ssd_tensorflow/master/images/Screenshot%20from%202016-11-02%2004%3A45%3A40.png" width="300"/>

Just looking through them, the results are okay but not good enough.

However, there are still major things needed to do that was done in the original paper for COCO but not here:

* Train on 500x500 images (this was 300x300)
* Use COCO trainval (this was only train)
* Use batch size 32 (this was only 8)

Other major improvements needed:

* Implement proper evaluation (mAP)
* Optimize training (currently pretty slow)

Concerns:

* Simple momentum optimizer stopped working (stopped converging) at some point during development, but adding batch normalization made it work again 

## Dependencies

* TensorFlow
* OpenCV

## Basic Instructions

1. You need vgg16.npy from [this repository](https://github.com/machrisaa/tensorflow-vgg), which is what I used for the base network. Unfortunately, it's a big file, and for now it's just uploaded to Mega, so you might need an account. [Direct link to npy](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM)

2. For now, the code uses and depends on MS COCO. You need the MS COCO dataset from [here](http://mscoco.org/dataset/#download). You should at least have the 2014 training images and corresponding 2014 train/val object instance annotations.

3. Download and install COCO tools from [here](https://github.com/pdollar/coco)

4. Change the COCO paths in coco_loader.py

5. Test or train
