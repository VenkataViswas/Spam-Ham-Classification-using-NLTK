# 📊 Text Classification using NLTK & Machine Learning

This project demonstrates a complete workflow for **text classification** — including data cleaning, preprocessing, feature extraction, and model building — using Python and machine learning techniques.

---

## 🧠 Project Overview
The goal of this project is to classify text messages into two categories:
- spam
- ham 

The model learns to identify patterns in text to make accurate predictions on unseen data.

---

## 🧹 Data Cleaning and Preprocessing

The following preprocessing steps were applied to the text data:

1. **Removing special characters and digits** using regular expressions  
2. **Converting text to lowercase**  
3. **Tokenization** – splitting sentences into words  
4. **Stopword removal** – removing common words (like “the”, “is”, etc.)  
5. **Lemmatization** – converting words to their base form (e.g., “running” → “run”)  

Example code snippet:
```python
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

corpus = []
for text in messages['message']:
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
    corpus.append(' '.join(review))
```

---

## 🔢 Feature Extraction
Converted the cleaned text into numerical vectors using **Bag of Words Vectorization**.

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500,ngram_range=(1,2))
```

---

## 🧩 Model Building

A supervised classification model was trained on the preprocessed text.  
Typical algorithms used for such tasks include:
- Logistic Regression  
- Naive Bayes  
- Support Vector Machine (SVM)  
- Random Forest


  As Naive Bayes algorithm works well for email classification we have used it 
---

## 📈 Model Performance

| Metric | Score |
|---------|-------|
| **Accuracy** | **0.979** |
| **Precision (Class 0)** | 0.99 |
| **Recall (Class 0)** | 0.99 |
| **F1-score (Class 0)** | 0.99 |
| **Precision (Class 1)** | 0.94 |
| **Recall (Class 1)** | 0.90 |
| **F1-score (Class 1)** | 0.92 |

**Classification Report:**
```
              precision    recall  f1-score   support
           0       0.99      0.99      0.99       968
           1       0.94      0.90      0.92       147
    accuracy                           0.98      1115
   macro avg       0.96      0.95      0.95      1115
weighted avg       0.98      0.98      0.98      1115
```

---

## 🧾 Results Summary
- The model achieves **97.9% accuracy** on the test dataset.  
- High **precision and recall** indicate strong performance for both classes.  
- The slight drop in recall for class `1` suggests a few false negatives.

---

## ⚙️ Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

Example packages:
```
numpy
pandas
scikit-learn
nltk
```

---

## 🚀 How to Run
1. Open the notebook:  
   ```bash
   jupyter notebook Main.ipynb
   ```
2. Run all cells in order.
3. Check the model performance output at the end.

---

## 🏁 Conclusion
The model performs exceptionally well in classifying text messages, achieving a test accuracy of **97.9%**.  

---


**Author:** *Venkata Viswas Tumbali*  
**Tooling:** Python 🐍 | scikit-learn | NLTK | Jupyter Notebook
