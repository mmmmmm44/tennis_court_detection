# Tennis Court Detection

Python Implementation of the paper "Robust Camera Calibration for Sport Videos using Court Models"

Development Branch.

## Log

Fixed some paramters in _numerical_dfferentiation()_ of _LMA.py_ so that the projected court model will not shift drastically within a few frames.

Discovered that the matrix M contains some extremely small value (~ 1e-16) that the original value will drastically change the rmserror, and the projection matrix H = inv(M), thus causing the drastic shift observed previously.

Modified error handling of singular matrix in _LMA.py_. Previously, it will shut the program down. However, I forcefully return the most up-to-date params, which is M, so that the program can continue.

Small fix in the projection error function to match the function _LM()_  of _LMA.py_
