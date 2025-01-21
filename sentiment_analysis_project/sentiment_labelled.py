import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

dataset_file = 'yelp_labelled.txt'  

data = pd.read_csv(dataset_file, sep='\t', header=None, names=['comment', 'label'])

X_train, X_test, y_train, y_test = train_test_split(
    data['comment'], data['label'], test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('nb', MultinomialNB())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2f}")
print("Relatório de classificação:\n", classification_report(y_test, y_pred))
