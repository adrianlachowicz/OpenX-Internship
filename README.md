# OpenX-Internship
## Machine Learning Project: CoverType Classification

This is a machine learning project that focuses on classifying cover types using the CoverType dataset. The dataset contains information about forest cover types, including soil types, elevation, and vegetation data.

The project includes the implementation of four different models, including a heuristics-based model, KNN, decision tree, and neural network. The models were trained and tested on the CoverType dataset. The whole project based on Python scripts.

The user interface for the project is provided through Gradio, a user-friendly Python library for creating customizable UI components. The UI allows the user to input data and receive predictions from the trained models.

To get started with the project, you can download and run a Docker image with the following command:

```docker run -p 8080:8080 adi282123/open_x_internship_project```

This will launch the application and make it available on your localhost at port 8080.

### Requirements
To run the project locally, you will need to have the following installed (all is in the ```requirements.txt``` file):

 - Python 3.7 or higher
 - Numpy
 - Pandas
 - Scikit-Learn
 - Gradio
 - Tensorflow
 - Keras
 - Optuna
 - Seaborn
 - Matplotlib
 
### Dataset
The CoverType dataset used in this project can be found [here](http://archive.ics.uci.edu/ml/datasets/covertype). It contains 581,012 rows and 55 columns of data. The dataset is divided into training and testing sets, with 70% of the data used for training and 30% for testing.

### Models
The following models were trained and tested on the CoverType dataset:

 - Heuristics-based model
 - KNN
 - Decision tree
 - Neural network

### User Interface
The user interface for the project is provided through Gradio. It includes input fields for the user to input data, and the predicted cover type is displayed as the output.

### License
This project is licensed under the MIT License - see the LICENSE file for details.
