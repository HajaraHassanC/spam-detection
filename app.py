import streamlit as st
import pickle

# 1. Load the existing model and vectorizer using pickle
with open("spam_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# 2. App Title
st.title("SMS Spam Detection")

# 3. Text area for entering the SMS message
user_input = st.text_area("Enter your SMS message:")

# 4. Predict Button
if st.button("Predict"):
    # 5. Handle empty input with a friendly message
    if not user_input.strip():
        st.warning("Please enter an SMS message first!")
    else:
        # 6. Transform the message using the loaded vectorizer
        data = vectorizer.transform([user_input])
        
        # 7. Predict using the loaded model
        prediction = model.predict(data)[0]
        
        # 8. Display either "Spam" or "Not Spam"
        if prediction.lower() == "spam":
            st.error("Spam")
        else:
            st.success("Not Spam")
