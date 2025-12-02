import streamlit as st
from predict import predict_sentiment

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.title("About")
    st.info(
        """
        This is a Sentiment Analysis Web App.
        - Built with Python, TensorFlow, and Streamlit.
        - Predicts Positive or Negative sentiment from text.
        """
    )
    st.image("https://i.imgur.com/1Q9Z1Z0.png", width=150)  # optional logo

# ---------------------------
# Header
# ---------------------------
st.title("📝 Sentiment Analysis App")
st.markdown(
    """
    Enter your text below to analyze its sentiment.  
    The model predicts whether the text is **Positive** 😄 or **Negative** 😢.
    """
)

# ---------------------------
# Input
# ---------------------------
text_input = st.text_area("Enter your text here:", height=150)

# ---------------------------
# Predict button
# ---------------------------
if st.button("Analyze Sentiment"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to analyze!")
    else:
        sentiment, confidence = predict_sentiment(text_input)
        label = "Positive 😄" if sentiment == 1 else "Negative 😢"

        # Color-coded result
        if sentiment == 1:
            st.success(f"Sentiment: {label}")
        else:
            st.error(f"Sentiment: {label}")

        # Confidence bar chart
        st.subheader("Prediction Confidence")
        st.bar_chart({"Positive": [confidence], "Negative": [1 - confidence]}, use_container_width=True)

        # ---------------------------
        # Prediction history
        # ---------------------------
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append((text_input, label, confidence))

        st.subheader("Last 5 Predictions")
        for idx, (txt, lbl, conf) in enumerate(reversed(st.session_state.history[-5:])):
            st.write(f"{idx + 1}. **{lbl}** — \"{txt}\" (Conf: {conf:.2f})")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("© 2025 Sentiment Analysis App | Built with Streamlit & TensorFlow")
