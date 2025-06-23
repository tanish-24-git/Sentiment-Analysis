Sentiment Analysis on Twitter Data
Overview
This repository contains a sentiment analysis project that classifies Twitter data into positive, negative, or neutral sentiments using logistic regression with TF-IDF features. Two implementations are provided:

Custom Pipeline (logics.ipynb): A from-scratch implementation using NumPy, with manual TF-IDF computation and logistic regression via gradient descent. Ideal for learning NLP fundamentals.
Scikit-Learn Pipeline (sklearn_logics.ipynb): A production-ready pipeline leveraging scikit-learn’s LogisticRegression and TfidfVectorizer, enhanced with NLTK for advanced preprocessing (lemmatization, tokenization). Includes cross-validation and hyperparameter tuning.

The project compares these pipelines, achieving 62.65% accuracy for the custom pipeline and 69.84% for the scikit-learn pipeline, demonstrating the trade-offs between learning and efficiency.
Repository Structure
Sentiment-Analysis/
│
├── dataset/                          # Twitter sentiment datasets (not tracked in Git due to size)
├── logics.ipynb                      # Custom sentiment analysis pipeline
├── sklearn_logics.ipynb              # Scikit-learn sentiment analysis pipeline
├── .gitignore                        # Ignores large datasets and cache files
└── README.md                         # Project documentation

Note: Large dataset files (e.g., training.1600000.processed.noemoticon.csv) are not included in the repository due to GitHub’s 100 MB file size limit. Instructions for downloading or sampling datasets are provided below.
Datasets
The project uses multiple Twitter sentiment datasets, including:

training.1600000.processed.noemoticon.csv (subsampled to 10% for efficiency).
Other datasets with text and sentiment labels.

Source: Datasets are publicly available (e.g., from Kaggle or Sentiment140). To replicate:

Download training.1600000.processed.noemoticon.csv from Sentiment140.
Place it in the dataset/ folder.
Alternatively, preprocess a smaller sample:import pandas as pd
df = pd.read_csv("dataset/training.1600000.processed.noemoticon.csv", encoding='latin-1')
df_sample = df.sample(frac=0.1, random_state=42)
df_sample.to_csv("dataset/training_sampled.csv", index=False)



Note: Large datasets are ignored via .gitignore. Use Git LFS if tracking large files.
Installation
Clone the repository and set up the environment.

Clone the Repository:
git clone https://github.com/tanish-24-git/Sentiment-Analysis.git
cd Sentiment-Analysis


Install Dependencies:Create a virtual environment and install required packages:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

If requirements.txt is unavailable, install manually:
pip install numpy pandas scikit-learn nltk matplotlib seaborn jupyter


Download NLTK Resources:Run the following in Python to download NLTK data:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


Prepare Datasets:Place datasets in the dataset/ folder or use a sampled version as described above.


Usage
Run the Jupyter notebooks to train, evaluate, and test the sentiment analysis models.

Start Jupyter Notebook:
jupyter notebook


Run Notebooks:

Open logics.ipynb for the custom pipeline.
Open sklearn_logics.ipynb for the scikit-learn pipeline.
Execute cells sequentially to:
Load and preprocess data.
Extract TF-IDF features.
Train logistic regression models.
Evaluate performance (accuracy, precision, recall, F1-score).
Test predictions on custom text (e.g., "I love this movie!").




Example Prediction:In sklearn_logics.ipynb, test a custom input:
text = "I love this movie, it's absolutely fantastic!"
# Output: Positive (86.47% probability)



Key Findings

Performance Comparison:
Custom Pipeline:
Test Accuracy: 62.65%
Macro F1-Score: 47.30%
Strengths: Transparent, educational, no external ML dependencies.
Weaknesses: Lower accuracy, memory-intensive, no regularization.


Scikit-Learn Pipeline:
Test Accuracy: 69.84% (+7.19%)
Macro F1-Score: 63.45% (+16.15%)
Strengths: Efficient, robust (cross-validation, grid search), NLTK preprocessing.
Weaknesses: Moderate accuracy, class imbalance issues.




Insights:
Scikit-learn’s optimized tools and advanced preprocessing (lemmatization) significantly improve performance.
Both pipelines struggle with neutral class recall due to data skew (negative class dominates).
The custom pipeline is ideal for learning NLP mechanics, while scikit-learn is production-ready.



Recommendations

Address Class Imbalance: Use SMOTE, oversampling, or class weights.
Enhance Features: Include n-grams or word embeddings (e.g., GloVe).
Try Advanced Models: Experiment with SVM, XGBoost, or transformers (BERT).
Optimize Datasets: Preprocess large datasets locally to reduce size.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/YourFeature).
Commit changes (git commit -m "Add YourFeature").
Push to the branch (git push origin feature/YourFeature).
Open a pull request.

Please ensure datasets are not committed directly due to size limits. Use Git LFS or provide preprocessing scripts.
License
This project is licensed under the MIT License. See LICENSE for details.
Contact
For questions or feedback, reach out via:

GitHub: tanish-24-git
LinkedIn: Your LinkedIn Profile (update with your profile URL)

Happy analyzing sentiments! 😊
