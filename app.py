import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Initialize the Porter Stemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = [word for word in y if word not in stopwords.words('english')]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Load or fit the model
try:
    with open('model_pipeline.pkl', 'rb') as f:
        model_pipeline = pickle.load(f)
except FileNotFoundError:
    expanded_texts = [
        "Congratulations! You've won a $1,000 gift card. Click the link to claim it now!",  # Spam
        "URGENT! Your account has been compromised. Verify your identity immediately to avoid suspension.",  # Spam
        "Get cheap meds online! No prescription needed. Click here to order now.",  # Spam
        "Dear customer, your loan has been pre-approved. Apply now and get instant cash!",  # Spam
        "Hi, just wanted to remind you about the meeting tomorrow at 10 AM.",  # Not Spam
        "Please find the attached report for your review. Let me know if you have any questions.",  # Not Spam
        "Thank you for your purchase! Your order has been shipped and will arrive soon.",  # Not Spam
        "Can we reschedule our lunch for next week? I have a conflict on Friday.",  # Not Spam
        "Limited time offer! Get 70% off on your next purchase. Click here to buy now!",  # Spam
        "Win a brand new iPhone by entering our free giveaway! Click here to enter.",  # Spam
        "Don't miss out on this exclusive offer! Buy now and save big!",  # Spam
        "Reminder: Your appointment with Dr. Smith is tomorrow at 3 PM.",  # Not Spam
        "Your order #12345 has been shipped and will be delivered by Friday.",  # Not Spam
        "Let's catch up sometime soon!",  # Not Spam
        "You have a new message from your bank.",  # Not Spam
        "Exclusive deal just for you!",  # Spam
        "See you at the meeting next week!"  # Not Spam
    ]
    expanded_labels = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1]  # 1 = Spam, 0 = Not Spam

    model_pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model_pipeline.fit(expanded_texts, expanded_labels)

    with open('model_pipeline.pkl', 'wb') as f:
        pickle.dump(model_pipeline, f)

# Streamlit app with custom styles
st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.markdown("""
    <style>
    body {
        background-color: #2c2c2c;  /* Darker background */
        color: #ffffff;  /* Default text color */
    }
    h1 {
        color: black;  /* Title in black */
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: #3a3a3a;  /* Sidebar background */
    }
    textarea {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        width: 100%;
        background-color: #444;  /* Dark textarea */
        color: #ffffff;  /* Text color in textarea */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
    }
    textarea:hover {
        border-color: #45a049;
    }
    .result {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📩 Email/SMS Spam Classifier")
st.markdown("This application classifies messages as Spam or Not Spam. Enter the message you want to analyze below:")

input_sms = st.text_area("Enter the message", height=150, placeholder="Write your message here...")

if st.button('Predict'):
    if input_sms:
        transformed_sms = transform_text(input_sms)
        result = model_pipeline.predict([transformed_sms])[0]

        if result == 1:
            st.markdown('<p class="result" style="color: red;">🚫This message is Spam</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="result" style="color: green;">✅ This message is Not Spam</p>', unsafe_allow_html=True)

        st.markdown(f"**Original Message:** {input_sms}")
        st.markdown(f"**Prediction Result:** {'Spam' if result == 1 else 'Not Spam'}")
    else:
        st.warning("Please enter a message to analyze.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
