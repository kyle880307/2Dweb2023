# DDW 2D Web App

This Web App explores predicting food security using *Multiple Linear Regression*.
  
The region we have chosen the Sub-Saharan African region as that region has the world's highest rate of malnutrition. Using metrics that would reflect the factors affecting food insecurity, we hope to predict where to send food to as a Humanitarian Food Aid NGO.

# How to use

To clone and run this application, you'll need Git installed on your computer. Or you can download the project folder aand open in Vscode or Anaconda From your command line:

```bash
# Clone this repository
$ git clone https://github.com/kyle880307/2Dweb2023.git

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


To start the program, the user should open an Anaconda Prompt/Vscode terminal and navigate to the directory of app folder and run the following command:

```

flask run

```

Alternatively, the user can use `python main.py` to run the program.

Once complete, the terminal shall output the following lines:


```

* Running on http://127.0.0.1:2020

Press CTRL+C to quit

* Restarting with watchdog (windowsapi)

* Debugger is active!

* Debugger PIN: XXX-XXX-XXX [This is a randomly generated string of digits]

```

The website can be accessed by opening the above URL in a web browser. There is a Navigation Bar containing links to pages in the order that the user is intended to use them.

1. Firstly, the user navigates to the 'Upload' page to download the template.csv and upload a CSV file containing the data that they wish to predict.


2. Once uploaded, the user can click 'Show CSV' to view the predictions by the model. The user will be redirected to the 'Predict' Page to view the predictions.

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
# Pages
  
## Overview Page
![enter image description here](https://i.imgur.com/sE5gRhN.png)
  
This is the landing page where the user first arrives. It gives  a brief introduction behind the motivation for creating the Web App.
## Upload Data

  ![enter image description here](https://i.imgur.com/ssZWG9J.png)
This is where the user is able to upload a CSV file for the user to be able to build a model.
## Predict
![enter image description here](https://i.imgur.com/qmGPvLr.png) This is where the user is able to make interpret predictions based on the training data.
## About Page
![enter image description here](https://i.imgur.com/WLwunhW.png)
