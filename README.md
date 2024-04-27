# Deep Learning Final Project

There is a requirements.txt file that lists all of the python requirements.

There will also probably be an ffmpeg dependency for the ahead-of-time pipeline.

* https://sites.google.com/view/utarldd/home
* https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset

* https://ieeexplore.ieee.org/document/8608130
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10384496/
* https://www.mdpi.com/2313-576X/9/3/65
* https://wiprotechblogs.medium.com/video-classification-for-drowsiness-detection-21a32b6f2ee0
* https://arxiv.org/pdf/2207.12148v1.pdf

# Data pipeline

Take the raw participant video and turn it into x, y training examples.

We will run the ahead-of-time pipeline before training. When it generates all of the data we need, we can use the just-in-time pipeline (`tf.Dataset`) during training.

## Ahead-of-time pipeline

http://dlib.net/face_landmark_detection.py.html

https://docs.opencv.org/4.x/d2/d42/tutorial_face_landmark_detection_in_an_image.html

https://learnopencv.com/facemark-facial-landmark-detection-using-opencv/

https://pyimagesearch.com/2018/04/02/faster-facial-landmark-detector-with-dlib/

http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2

https://ffmpeg.org/ffmpeg-filters.html#perspective

Here is how to build the data from the raw participant videos. The repository will include the final processed video, so you probably don't need to do this.

* Make sure `data/init/` contains the raw participant videos from kaggle. You can make `data/init/` a symlink to an external or network drive if you need to. Because of its large size, procuring `data/init/` is your responsibility, and no recipe is provided for these files.

* Run `src/partipants.py`. This will generate `data/listing.txt` from `data/init/`. It is a list of the partipant ids. This file is in verison control because it can be useful to know the partipant ids withot the raw participant videos.

* Run `src/split.py`. This generates `data/{train,test,validation}.txt` from `data/partipants.txt`, which each contain the participant ids in their respective split. The files generated belong in version control, since the train/test split should be stable. This can be run anytime, even after building the data pipeline.

* Run `src/makefile.py` to generate the `Makefile`. This script will inspect `data/listing.txt`, and create a `Makefile` with the recipes to build the ahead-of-time data files. Each data file is its own target. The generated `Makefile` will be in version control because it contains the mapping from source video files to participant ids.

* Run `make -j8`. This will build out the data from `data/init/`. This command can be interrupted arbitrarily and is expected to take very long. The table has info about the resources generated.

resource | source | description
---|---|---
`data/init/[participant id]/{0,5,10}.{mp4,mov}` | kaggle | the raw participant's videos downloaded directly from kaggle
`data/positions/[participant id]/{0,5,10}.csv` | `data/init/` | the position of keypoints on the particpant's eyes for each frame
`data/extract/[participant id]/{0,5,10}.mp4` | `data/positions/`, `data/init/` | applying a projection to every frame to extract videos of just the participant's eyes

## Just-in-time pipeline

These things can all happen in tensorflow during training.

https://www.tensorflow.org/api_docs/python/tf/data/Dataset

https://www.tensorflow.org/tutorials/load_data/video

The just-in-time pipeline should preserve the video size. Also,
the position of the eyes should be in approximately the same place in every
video, because that's the job of the ahead-of-time pipeline. I anticipate that
the eyes won't be in exactly the same spot, so we can use augmentation allow
the model to be comfortable with the inevitable small shifts.

* Take videos from the ahead-of-time pipeline
* split video into small clips. Small enough for the model to be able to process the entire clip, but large enough to see the time-series detail that we need to make the drowsy classification.
* augmentation - color shifts, small noise addition, jpeg re-encode, light blur, tiny rotations, flip left-right, tiny rescale 
* shuffle video clips. This may be tricky because we want to randomize across all files, but can't read the entire dataset into memory. Maybe this should be the ahead-of-time pipeline's job?
* batching 

##  Getting the Vision Transformer submodule's code
https://gist.github.com/gitaarik/8735255
If a new submodule is created by one person, the other people in the team need to initiate this submodule. First you have to get the information about the submodule, this is retrieved by a normal git pull. If there are new submodules you'll see it in the output of git pull. Then you'll have to initiate them with:
`git submodule init`
