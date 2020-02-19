# Advanced Machine Learning TAU
This repository is part of the course: *Advanced Machine Learning* in Tel Aviv University.
The aim here is to check whether the theoretical ![formula](https://render.githubusercontent.com/render/math?math=\lambda) value will coincide with the ![formula](https://render.githubusercontent.com/render/math?math=\lambda) that gave the best results.

A document describing the process is present.

## Implementation
The code was implemented using `python 3.x`.
It was tested on:
* Mac OS - Mojave
* Ubuntu - 18.04 LTS
* Windows - 10

## Dependencies
* `sklearn`
* `numpy`
* `copy`
* `cvxopt`
* `matplotlib`
* `mlxtend`
* `os`
* `time`

## Run
* Edit the appropriate block in the **Main.py** wrapped with "########## EDIT ##########"
where one should choose the desired classes\ matrix covariance. For the Mnist, Iris & Wine datasets the classes are (0-9) & (0-2) respectively.
As for the matrix covariance, notice to keep the matrix as a diagonal one for simplicity.
* Then run `python3 Main.py` in the terminal/cmd
* Besides the output on the screen describing the values of ![formula](https://render.githubusercontent.com/render/math?math=\lambda) the script output plots
in .eps format directly to "images" directory {in the case it does not exist it creates it}
