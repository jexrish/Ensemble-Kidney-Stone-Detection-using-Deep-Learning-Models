Overview  
This project introduces a hybrid machine learning model that detects kidney stones in CT scan images. It takes advantage of pre-trained deep learning models for automatic feature extraction and combines them with a traditional machine learning ensemble for strong classification.

The main steps include:

- Extracting deep features from CT images using ResNet101 and InceptionV3.
- Combining these features to form a rich, hybrid feature vector.
- Applying PCA to reduce dimensions.
- Training an Ensemble Voting Classifier, which uses SVM and KNN, on the processed features to label images as "Healthy" or "Kidney Stone".

The entire process is contained within a Jupyter Notebook (proj.ipynb), designed for use on Google Colab.

Features  
Hybrid Model: Merges deep learning feature extraction with a traditional machine learning classifier.  
Feature Engineering: Uses pre-trained ResNet101 and InceptionV3 models to automatically identify important features from images.  
Ensemble Learning: Implements a Voting Classifier with Support Vector Machine and K-Nearest Neighbors to enhance prediction stability and accuracy.  
Hyperparameter Tuning: Applies GridSearchCV to determine the best parameters for the ensemble model.  
Performance Evaluation: Offers a thorough analysis of the model's performance, including a classification report, confusion matrix, and ROC curve with AUC score.  
Model Persistence: Saves the final trained model using joblib for future predictions.  
Reproducibility: Stores test set predictions in a CSV file for further analysis.

Tech Stack  
Python 3.x  
TensorFlow & Keras: For loading pre-trained models and processing images.  
Scikit-learn: For machine learning pipelines, models (SVM, KNN), evaluation metrics, and PCA.  
Pandas & NumPy: For data manipulation and numerical tasks.  
Matplotlib & Seaborn: For data visualization.  
OpenCV: For image preprocessing.  
Google Colab / Jupyter Notebook: As the development environment.

Dataset  
This project uses the CT Kidney Dataset: Cysts, Tumors, and Stones, which is publicly available on Kaggle.  
Source: Kaggle Dataset Link

Getting Started  
Follow these instructions to set up and run the project on your local machine or in Google Colab.

Prerequisites  
- Python 3.8+  
- pip package manager  

1. Clone the Repository  
2. Set up the Dataset  
Download the dataset from the Kaggle link provided above.  

Unzip the file and create a directory structure as shown in the Dataset section. For this project, you only need the folders named Stone and Normal.  

Rename the folders to kidney_stone and healthy respectively to match the script.  

If using Google Colab, upload this KidneyCTDataset folder to your Google Drive.  

3. Install Dependencies  
It is recommended to use a virtual environment.  
4. Running the Project  
Open the proj.ipynb notebook in Google Colab or a local Jupyter Notebook environment.  
Run all the cells in the notebook one after the other from top to bottom.

Results  
The ensemble model performed well on the test set, showcasing the effectiveness of the hybrid approach.

Future Improvements  
End-to-End Fine-Tuning: Rather than only using pre-trained models as feature extractors, unfreeze some of the top layers and fine-tune them on the CT scan data.  
Advanced Models: Try more modern architectures like EfficientNet or Vision Transformer (ViT).  
Web Application: Create a simple web interface using Flask or Streamlit where users can upload a CT scan image and receive an immediate prediction.  
Explainable AI (XAI): Apply techniques like Grad-CAM to visualize which parts of the image the model focuses on, making the predictions easier to understand.
