# Deep Learning Final Project

There is a requirements.txt file that lists all of the python requirements.

* https://sites.google.com/view/utarldd/home
* https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset

* https://ieeexplore.ieee.org/document/8608130
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10384496/
* https://www.mdpi.com/2313-576X/9/3/65
* https://wiprotechblogs.medium.com/video-classification-for-drowsiness-detection-21a32b6f2ee0
* https://arxiv.org/pdf/2207.12148v1.pdf

# Data pipeline

Take the raw participant video and turn it into x, y training examples.

## The ahead-of-time data pipeline

resource | source | description
---|---|---
`data/init/*/*/*/{0,5,10}.{mp4,mov}` | kaggle | the raw participant's videos downloaded directly from kaggle
`data/flat/[participant id]/{0,5,10}.{mp4,mov}` | `data/init/` | get a participant id for each person
`data/positions/[participant id]/{0,5,10}.csv` | `data/flat/` | the position of keypoints on the particpant's eyes for each frame
`data/extract/[participant id]/{0,5,10}.mp4` | `data/positions/`, `data/flat/` | applying a projection to every frame to extract videos of just the participant's eyes
`data/split/{train,test,validation}/[partipant id]/{0,5,10}.mp4` | `data/extract/` | split the data into train, test, validation based on the partipant id

We can keep `data/extract` and `data/split` in version control. `data/split` and `data/flat` should be symlinks to their respective sources.

It would be nice to use `make` for this entire pipeline so that we could take advantage of conditional re-building, and `-j8` parallelism.

http://dlib.net/face_landmark_detection.py.html

https://docs.opencv.org/4.x/d2/d42/tutorial_face_landmark_detection_in_an_image.html

https://learnopencv.com/facemark-facial-landmark-detection-using-opencv/

https://pyimagesearch.com/2018/04/02/faster-facial-landmark-detector-with-dlib/

http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2

### Building the ahead-of-time data

The repository will include the final processed video, so you probably don't need to do this.

If you would like to download and rebuild everything from the raw dataset, run

```sh
make -B data
```

This will use several hundred gigabytes of disk space, and take many hours.

## Just-in-time pipeline

These things can all happen in tensorflow just-in-time.

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

