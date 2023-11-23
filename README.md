# 2Dweb2023
# DDW 2D Web App

  

This Web App explores predicting food security using *Multiple Linear Regression*.

  

The region we have chosen the Sub-Saharan African region as that region has the world's highest rate of malnutrition. Using metrics that would reflect the factors affecting food insecurity, we hope to predict where to send food to as a Humanitarian Food Aid NGO.

  

# Packages used

Download the whole zip file and go to the root directory of the project.

  

## Pip installation

```
pip install Flask
pip install numpy
pip install pandas
pip install Werkzeug
```

  

# How to use

To clone and run this application, you'll need Git and Node.js (which comes with npm) installed on your computer. From your command line:

```bash
# Clone this repository
$ git clone 

# Go into the repository
$ cd app

# Install dependencies
$ pip install Flask
$ pip install numpy
$ pip install pandas
$ pip install Werkzeug

# Run the app
$ flask run
```

To start the program, the user should open an Anaconda Prompt terminal and navigate to the directory of `main.py` and run the following command:


```
python main.py
```

Alternatively, the user can use `flask run` to run the program.

  

```
flask run
```
Once complete, the terminal shall output the following lines:
```
* Running on http://127.0.0.1:2020
Press CTRL+C to quit
* Restarting with watchdog (windowsapi)
* Debugger is active!
* Debugger PIN: XXX-XXX-XXX [This is a randomly generated string of digits]
```

The website can be accessed by opening the above URL in a web browser. There is a Navigation Bar containing links to pages in the order that the user is intended to use them.

1. Firstly, the user navigates to the 'Upload' page to upload a CSV file containing training data that they wish to build a model from.

- The first column of the CSV file must be indices

- The last 2 columns of the CSV file must be the target data

- The columns in between must be the feature data

- For testing purposes, the user can upload the provided csv file named `dataset_testtrain_V2.csv`

2. Once uploaded, the user can click 'Show CSV' to view the predictions by the model. The user will be redirected to the 'Excel' Page to view the predictions.

- If there are more than 10 indices, the program will automatically paginate the predictions.
 
  

# File Structure

```  
app/
├── app.py
├── predict.py
├── static/
│   ├── csv
│   ├── data
│   └── images
└── templates/
	├── about.html
	├── base.html
	├── excel.html
	├── index.html	
	├── input.html
	└── input2.html
```

  

# Features

  
  
  

## About page (welcome)

  

## Build Model

  

## Predict
