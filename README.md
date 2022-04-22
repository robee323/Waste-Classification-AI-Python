# Waste Classification application using Artificial Intelligence in Python
Project for Biologically inspired artificial intelligence subject - Waste Classification application using Artificial Intelligence

# Contributors
Robert Lotawiec
Adam Gembala

# Subject of the project
The aim of the project is to create an AI model that will analyse images passed by the user and then identify waste as belonging to one of the categories. The artificial intelligence model will learn to recognize waste category based on our data set. The categories included in the data set: cardboard, glass, paper, metal, plastic and unclassified garbage.

# Overview
The project will be implemented in Python with the use of appropriate libraries and cooperating tools. The user will enter in the console the path to the selected photo to be analysed. The analysis of the photo will consist of extracting objects from the photo and then identifying them according to the category of objects from the data set. The result of executing the user's command will be text feedback on the identified objects on the console.

# Method of implementation
We will use detection to interpret and identify the content of images. Detection is a computer technology related to image processing that deals with detecting instances of semantic objects of a certain class. Every object class has its own special features that helps in classifying the class - for example all circles are round. Object class detection uses these special features. Based on the available waste categories, we will want to formulate characteristic attributes for each waste class.

# Sources (data sources, solutions used, libraries, etc.)
Along with Python as the programming language we will use the OpenCV library. We decided to use this library because it is commonly recommended for object detection projects. We will be using NumPy library for matrix computation and Matplotlib for visualization of the output image.
Our dataset consists of images divided into six categories: cardboard, glass, metal, paper, plastic and trash. We will split each category into training (~80%) and test (~20%) sets.
Link to dataset: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification?datasetId=81794

