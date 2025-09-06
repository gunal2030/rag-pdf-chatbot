import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import time

# --- Configuration ---
# Set your API key in Streamlit secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Failed to configure API key: {e}")
    st.stop()

# Initialize the Gemini model
# We'll use the 'gemini-pro-vision' model, which is multimodal
model = genai.GenerativeModel('gemini-pro-vision')

# Set up the Streamlit app page
st.set_page_config(
    page_title="Multimodal Q&A with Gemini",
    page_icon="🤖"
)
st.title("🤖 Ask a question about your image")

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_image_name" not in st.session_state:
    st.session_state.uploaded_image_name = None

# --- Helper Functions ---
def get_gemini_response(prompt_parts):
    try:
        start_time = time.time()
        response = model.generate_content(prompt_parts)
        end_time = time.time()
        st.sidebar.markdown(f"**Response Time:** {end_time - start_time:.2f} seconds")
        return response.text
    except Exception as e:
        return f"An API error occurred: {e}"

# --- UI Elements ---
with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    st.info("A higher temperature makes the answer more creative.")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# --- Main App Logic ---
if uploaded_file is not None:
    # Check if a new image was uploaded
    if st.session_state.uploaded_image_name != uploaded_file.name:
        st.session_state.uploaded_image_name = uploaded_file.name
        st.session_state.messages = []  # Clear history for new image

    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Add "What's in the picture?" button
    if st.button("What's in the picture?"):
        st.session_state.messages.append({"role": "user", "content": "What's in the picture?"})
        with st.chat_message("user"):
            st.markdown("What's in the picture?")
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_gemini_response(["Describe the image in detail.", image])
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Show conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if user_prompt := st.chat_input("Ask a question about the image..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Get assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_gemini_response([user_prompt, image])
                st.markdown(response)

        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Please upload an image to begin.")