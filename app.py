import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, we embark on an interactive study of **phoneme characteristics** and their pivotal role in **Text-to-Speech (TTS) technologies**. Understanding the fundamental building blocks of speech—phonemes—and their measurable acoustic properties such as **duration ($\text{ms}$)**, **pitch ($\text{Hz}$)**, and **energy** is crucial for developing advanced TTS systems that produce natural, expressive speech, and for creating effective language learning tools.

This application provides a hands-on environment to:
*   **Generate and explore synthetic phonetic datasets**
*   **Visualize the distributions and relationships** between various phoneme features
*   **Simulate latent feature creation and apply clustering** to identify phoneme groupings
*   **Build a simple predictive model** for "pronunciation naturalness"
*   **Interact with individual phoneme data** to compare characteristics

Our aim is to shed light on how these intricate phonetic features collectively contribute to the naturalness and intelligibility of speech, paving the way for innovations in synthetic voice generation and language education.
""")
# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Data Generation & Validation", "Visualizing Relationships & Comparisons", "Advanced Analysis & Interactive Tools"])
if page == "Data Generation & Validation":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Visualizing Relationships & Comparisons":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Advanced Analysis & Interactive Tools":
    from application_pages.page3 import run_page3
    run_page3()
# Your code ends
