# Email Spam Detection System

## Overview

This project implements an Email Spam Detection System using Machine Learning to classify messages as Spam or Not Spam. The system leverages Natural Language Processing (NLP) techniques and a trained Naive Bayes classifier, deployed through an interactive Streamlit web application for real-time prediction.

## Features

🎯 **Machine Learning Powered**: Uses Naive Bayes for accurate text classification
⚡ **Real-time Prediction: Instantly classifies messages as spam or legitimate
🧠 **NLP-Based Processing**: TF-IDF vectorization for feature extraction
🖥️ **Interactive UI**: Simple and user-friendly Streamlit interface
📊 **High Accuracy**: Achieves ~95% accuracy on benchmark dataset
💾 **Model Persistence**: Pre-trained model and vectorizer stored as pickle files

## Project Structure
├── dataset/
│   └── spam.csv                     # SMS Spam Collection dataset
├── spam_model.pkl                   # Trained Naive Bayes model
├── vectorizer.pkl                   # TF-IDF vectorizer
├── streamlit_app.py                 # Streamlit web application
├── requirements.txt                 # Python dependencies
└── README.md                        # This file

## Technical Implementation
### Machine Learning Model

- **Algorithm**: Multinomial Naive Bayes
- **Text Representation**: TF-IDF Vectorization
- **Task**: Binary classification (Spam / Not Spam)
- **Performance: Achieves approximately 95% accuracy
- **Preprocessing**:Text normalization
Stopword removalFeature extraction using TF-IDF

### Data Pipeline

- **Data Source**: SMS Spam Collection Dataset
- **Dataset Size**: 5,500+ labeled messages
- **Labels**:
spam – Unwanted or fraudulent messages
ham – Legitimate messages
- **Encoding Handling**: Latin-1 encoding support
- **Data Cleaning**: Removal of unnecessary columns and normalization of column names

### Application Features

- **Home Page**: Project overview and instructions 
- **Spam Detection Interface**: Text input area for message entry
-**Prediction Engine**: Real-time classification

### Result Display:

❌ Spam Message

✅ Not Spam Message

## Installation & Setup

1. **Clone or Download the Project Files**
2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```
3. **Run the Streamlit Application**
```bash
streamlit run streamlit_app.py
```
4. **Access the Application**Open your browser and navigate to:
http://localhost:8501`

## 🚀 Live Demo

https://emailspamdetectionsystem.streamlit.app/

## Usage Guide
### 1. Spam Detection

-Enter email or message text into the input box
-Click “Check Spam”
-View instant classification result

### 2. Example Inputs
-Spam Example
-Congratulations! You have won a free iPhone. Click now!
-Not Spam Example
-Hi, are we meeting tomorrow at 10 AM?

## Model Performance
The trained model achieves the following metrics:
-**Accuracy**: ~95%
-**Precision**: High for spam class
-**Recall**: Effective spam detection
-**F1-Score**: Balanced classification

*Note: Performance may vary with real-world email data.*

## Key Concepts Explained
### Text Classification

The system converts raw text into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency), which reflects the importance of words relative to the dataset.

## Naive Bayes Algorithm

Naive Bayes is a probabilistic classification algorithm that performs exceptionally well for text-based problems due to its assumption of word independence.

## Spam Classification Logic

Spam (1): Promotional, fraudulent, or unwanted messages
Not Spam (0): Genuine and meaningful messages
Technology Stack
Language: Python
Machine Learning: scikit-learn
NLP: NLTK, TF-IDF
Frontend: Streamlit
Data Handling: pandas, numpy
Model Storage: Pickle
Deployment: Streamlit Cloud

## Future Enhancements

- Email header analysis
- URL and link-based spam detection
- Deep learning models (LSTM, Transformers)
- Multi-language spam detection
- API integration for email platforms

## License

This project is developed for educational and academic purposes.

## Contact

For questions or support, please refer to the code comments and documentation provided within the repository.