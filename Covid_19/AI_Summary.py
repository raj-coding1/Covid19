import streamlit as st
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyAV4-9fv3ltLsPKySpHeRTojdyzr_BXG_o")
model = genai.GenerativeModel("gemini-pro")

st.title("AI Generated Summary")

# You can pass your data summary or statistics here from session_state or re-process
sample_input = """
The global confirmed cases are 700 million, deaths 6.9 million, and recoveries 680 million.
India has reported 44 million confirmed cases, 43 million recoveries, and 530,000 deaths.
"""

if st.session_state.get('generate_summary'):
    with st.spinner("Generating summary..."):
        response = model.generate_content(f"Generate a beautiful short summary of this COVID data:\n{sample_input}")
        st.success("Summary Generated")
        st.markdown(response.text)
else:
    st.warning("Click 'AI Generated Summary' on the main page.")
