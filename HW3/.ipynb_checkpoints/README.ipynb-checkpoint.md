# Readme HW3

__Important:__ If you struggle with any part of this assignments, raise an issue in the repo (see the section below on how to submit)

## PART 1 

Create a notebook `HW3-part-1.ipynb` and do the following in it:

(1) Write a function which takes three inputs - (1) an HICP item code (a string), (2)  a boolean (True or False) indicating if the series is seasonally adjusted or not, and (3) an indicator whether the series is in levels or as annual rate of change (for example, the parameter can be a string with one of two options - "INX" or "ANR"). The function should download the indicated adjusted or unadjusted HICP series in level ("INX") or as rate of change ("ANR") and returns a dataframe with datetime index and one series whose column name shows the item description, geo, and whether it is seasonally adjusted or unadjusted.

__Note:__ You should use the function you wrote in HW2, making the necessary changes to allow for data in levels and with seasonal adjustments.

(2) Using the function from (1) download HICP in levels for U2, seasonally adjusted __and__ unadjusted

(3) Compute year on year inflation rates as per cent change from the previous year

(4) Compute year on year inflation rates as the 12-month difference in the log of the series

__Hint:__ [this](https://niskrev.github.io/PhDEcon108/02-Tutorials/02-Pandas/02-Pandas-for-time-series.html) may be helpful

(5) Using the function from (1) download HICP in annual rate of change for U2, seasonally unadjusted.

(6) Make a 2 by 2 plot showing __each one__ of the four inflation series you calculated in (3) and (4) (per cent change adjusted and unadjusted, log-diff adjusted and unadjusted) and the ANR series you downloaded in (5). 

__Hint:__ [check this out](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html)

(7) Which of the two methods you used produces inflation series that is closer to the official inflation data? 


## Part 2

(1) Read/work through the examples in [Numpy tutorials](https://niskrev.github.io/PhDEcon108/02-Tutorials/01-Numpy/00-Numpy.html)

(2) See notebook [HW3-part-2-2.ipynb](./HW3-part2-2.ipynb)

## Part 3

Write down what difficulties, questions, or points of confusion you had or still have about this or previous assignments.


## Submission
Submit as a folder with name __HW3__ in your repository at https://github.com/PhDEcon108. In addition to the the two notebooks, the folder should contain a README.md file describing the content of the repository. It should also have a section where you write your answer to Part 3. 

