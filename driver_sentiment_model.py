# Sentiment Analysis of Driver Feedback - ML Model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample driver feedback data
X = ["I feel sleepy", "I'm angry", "I'm okay", "I'm frustrated"]
y = ["negative", "negative", "positive", "negative"]

# Convert text to numeric
cv = CountVectorizer()
X_vec = cv.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Test with new input
test = ["I'm feeling fine"]
test_vec = cv.transform(test)
prediction = model.predict(test_vec)
print("Driver Sentiment:", prediction[0])
