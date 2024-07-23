import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Generate synthetic chat data
def generate_synthetic_data():
    data = {
        'question': [
            'How can I reset my password?',
            'What are the store hours?',
            'I want to return a product.',
            'Can you help me with my order?',
            'How do I track my shipment?'
        ],
        'answer': [
            'To reset your password, go to the account settings page and click "Reset Password".',
            'Our store hours are Monday to Friday, 9 AM to 6 PM.',
            'To return a product, please visit our returns page and follow the instructions.',
            'Please provide your order number, and we will assist you.',
            'You can track your shipment by entering your tracking number on our tracking page.'
        ]
    }
    df = pd.DataFrame(data)
    return df

df = generate_synthetic_data()

# Preprocess data
def preprocess_data(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['question'])
    y = df['answer']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, vectorizer

X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df)

# Save the vectorizer
import pickle
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
