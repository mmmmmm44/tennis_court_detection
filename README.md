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
