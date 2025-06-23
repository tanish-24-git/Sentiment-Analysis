Sentiment Analysis Comparison 😊😢😐

Performance Comparison 🗣️

Custom Pipeline (logics.ipynb) 🐍:

Test Accuracy: 62.65%
Macro F1-Score: 47.30%
Strengths: Transparent, educational, no external ML dependencies.
Weaknesses: Lower accuracy, memory-intensive, no regularization.


Scikit-Learn Pipeline (sklearn_logics.ipynb) 📊🛠️:

Test Accuracy: 69.84% (+7.19%)
Macro F1-Score: 63.45% (+16.15%)
Strengths: Efficient, robust (cross-validation, grid search), NLTK preprocessing.
Weaknesses: Moderate accuracy, class imbalance issues.



Insights 🔍

Scikit-learn’s optimized tools and advanced preprocessing (lemmatization) significantly improve performance 📈.
Both pipelines struggle with neutral class recall due to data skew (negative class dominates).
The custom pipeline is ideal for learning NLP mechanics, while scikit-learn is production-ready.
