# Tennis Court Detection

Python Implementation **UPTO Section 3** of the paper "Robust Camera Calibration for Sport Videos using Court Models"

For development about Section 4, please switch to branch "dev_la_algo"

![Result 02](result_samples/result_pic_02.png)

![Result 03](result_samples/result_pic_03.png)

![Result 08](result_samples/result_pic_08.png)

## Project Structure

### Programs

| File | Description |
| --- | --- |
| config.py | Constants for the algorithm in the paper. |
| court_line_candidate_detector.py | Implementation of Section 3.2 of the paper. |
| line.py | Self-defined line object for storing the equation, parameterized parameters. Also include implementation of drawing lines to cv images. |
| main_single_image.py | main program to run my implementation. |
| model_fitting.py | Implementation of Section 3.3 of the paper. |
| skimage_hough_transform.py | program for testing skimage hough transform function. |
| tennis_court_model.py | Definition of the tennis court model. |
| utils.py | Utility functions for drawing lines and court model on cv2 image, and resize an image.
| white_pixel_detector.py  | Implementation of Section 3.1 of the paper. |

### Others

| File | Description |
| --- | --- |
| Farin2004b_slides.pdf | The presentation slides of the paper. |
| Robust Camera Calibration for Sport Videos using Court Models.pdf | The paper itself. |

## Usage

Environment (simply list out major packages.)

- python = 3.9.9
- opencv = 4.5.5
- scikit-image = 0.19.3
- pillow = 9.1.1
- matplotlib = 3.5.2
- scipy = 1.8.1
- numpy

1. Create an environment and install the packages above

2. For single image inferencing, executing the following

```python
python main_single_image.py
```

## Future Developments

- [ ] Implement Section 4 of the paper.

## Why this paper/project?

Researching about court detection during an internship, I stumbled on this paper, and found that the algorithm and easy-to-understand metholodogy can be helpful for my work in the internship. Despite around 20 years after its release, I think implementing the algorithm can enhance my understand about homography and image processing, which are two major aspects in computer vision. It also sets a challenge for myself to understand academic research, which is a crucial ability in research.

Despite the seeming straight-forward steps, multiple unexpected difficulties arose during implementation. From finding a suitable implementation of LMedS estimator, or other similar estimator, to understanding Section 3.3, model-fitting, and Section 4, which is the most difficult section to implement as implementing gradient descent algorithms beyonds my capability. Understanding others' Python implementation and integrate it to my work is also demanding.

I hope this project demonstrates my fundamental understanding about computer vision and skills to implement them.
Python Implementation of the paper "Robust Camera Calibration for Sport Videos using Court Models", **both Section 3 and Section 4**

Development Branch.

![Result 05](result_samples/result_05.gif)

This branch is delicated for Section 4, as my implementation is not perfect. The projected court model will deviate bit by bit from the court lines, which affect the detection and selection of white pixels, causing a vicious cycle. The deviation can be observed after ~ 20 to 30 frames.

## Program Structure

Most of the files are shared with main branch, except the files below.

| File | Description |
| --- | --- |
| main_single_video.py | Script to run both Section 3 and Section 4 on a short video. It takes around 1 sec per frame. |

## Log

Fixed some paramters in _numerical_dfferentiation()_ of _LMA.py_ so that the projected court model will not shift drastically within a few frames.

Discovered that the matrix M contains some extremely small value (~ 1e-16) that the original value will drastically change the rmserror, and the projection matrix H = inv(M), thus causing the drastic shift observed previously.

Modified error handling of singular matrix in _LMA.py_. Previously, it will shut the program down. However, I forcefully return the most up-to-date params, which is M, so that the program can continue.

Small fix in the projection error function to match the function _LM()_  of _LMA.py_

## Insights

TLDR: fine tune the _closest_dist_ threshold in _main_video()_ of _main_single_video.py_, and both _delta_factor_ and _min_delta_ in _numerical_differentiation()_ of _LMA.py_ to improve the model

1. I observe that the most important step is the white pixel detection, which is section 3.1 of the paper. An accurate white pixel detector with as little noise as possible hugely increases the accuracy of the projection matrix. The subsequent steps are refinement of the projection matrix, using information such as distance between pixels and court lines, and rejecting impossible projection matrices. As the saying said, garbage in garbage out, the model will be accurate if the pixel detection algorithm is robust and accurate. Thus, playing around with the distance rejecting pixels too far away from the court model after inverse projection is instrumental to the model design.

2. The inverse of the projection matrix is extremely numeric sensitive. The value of the elements of the inverse can be as miniscule as 1e-16, which the index is 1/2 to the minimum of the typical float standard. Hence, playing around with the _delta_factor_ and _min_delta_ in _numerical_differentiation()_, and reduction threshold in _LA()_ of _LMA.py_ is crucial in fine-tuning the performance of the model.
