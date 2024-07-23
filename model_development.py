from sklearn.naive_bayes import MultinomialNB
import pickle

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model
with open('chatbot_model.pkl', 'wb') as file:
    pickle.dump(model, file)
