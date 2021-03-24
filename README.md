<div align="center">
  <img src="docs/images/design.png" style="width:80%"><br>
</div>

# Find Running Routes Matching Shapes
These scripts combine graph search and deep convolutional neural networks to suggest running routes that match a specified emoji / character shape in a given city. The current implementation uses a simple brute-force search approach to identify every closed path (subject to a few simple contraints) in the target area. Each closed path is scored by the convolutional neural network. High scoring candidates are stored in a CSV file for review.

# Installation
Install the dependencies in requirements.txt:

> pip install -r requirements.txt

Consider installing and running the tool in a virtual environment:

https://docs.python.org/3/library/venv.html

# build.py
Builds a neural network model to identify the specified character / emoji. Use this script first and note that _training time is heavily dependent on volume of training data (controlled by the size parameter)_. 

Note that emojis with simple geometry work the best.

> py build.py --character ❤

```
usage: build.py [-h] [--character CHARACTER] [--save SAVE] [--savedir SAVEDIR]
                [--size SIZE] [--model MODEL] [--city CITY] [--dim DIM]

[shapecat] Model Build Tool

optional arguments:
  -h, --help            show this help message and exit
  --character CHARACTER
                        character/emoji for model construction
  --save SAVE           save copies of training images
  --savedir SAVEDIR     training image storage location (only relevant if
                        save=true)
  --size SIZE           the number of positive and negative examples (each)
  --model MODEL         the name of the output model
  --city CITY           the name of the city for negative training data
  --dim DIM             the width/height of the training images
                        (larger=slower)
```

# search.py
Uses a trained neural network model (created with build.py) to track down shapes in your target geographic area. __Results are stored in a csv file (matches.csv by default).__ Supply a city in quotes. 

The __threshold__ parameter (defaulted to 0.5, accepts values between 0-1) controls how the minimum match amount needed to identify a given route as a match. Increase this parameteter (up to 0.99) to reduce the amount of false positives. Decrease this parameter to reduce the amount of false negatives.

The parameters __dmin__ and __dmax__ may be used to target routes that match a certain distance. The default values of 500 and 10000 will match routes falling in the range of 500-10000 meters. Note that many interesting (matching) shapes may fall above or below your ideal route length and it's a good idea to leave this range as large as possible.

The parameters __smin__ and __smax__ influence the complexity of the search by limiting the lower and upper bound of number of unique road/path segments to be considered. The initial graph search process uses these constraints to speed up the process. The upper bound, __smax__ (defaulted to 10), is the most influential on performance and small increases may result in a large non-linear increase in search time. Because not all road/path segments are straight, smaller values may still allow the search process to identify complex geometry. However, in general, increasing the upper bound will allow the search to find more complex shapes at the cost of speed. 

> py search.py "St. Petersburg, FL, USA"

```
usage: search.py [-h] [--threshold THRESHOLD] [--model MODEL]
                 [--result_csv RESULT_CSV] [--result_dir RESULT_DIR]
                 [--smin SMIN] [--smax SMAX] [--dmin DMIN] [--dmax DMAX]
                 [--visualize VISUALIZE]
                 city

[shapecat] Search Tool

positional arguments:
  city                  the name of the location to search in

optional arguments:
  -h, --help            show this help message and exit
  --threshold THRESHOLD
                        threshold for matching images (0-1)
  --model MODEL         the name of the input model
  --result_csv RESULT_CSV
                        the name of the results file to store matches
  --result_dir RESULT_DIR
                        the name of the results directory to store matches
  --smin SMIN           the minimum number of segments to match
  --smax SMAX           the maximum number of segments to match
  --dmin DMIN           the minimum distance to match, in meters
  --dmax DMAX           the maximum distance to match, in meters
  --visualize VISUALIZE
                        controls image saving to [result_dir] directory
```

# Running the Unit Tests
A small selection of unit tests exercise the graph search portion of the application through the candidate finder class. To run the tests, use pytest as follows:

> pytest tests