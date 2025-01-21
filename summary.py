# Install required libraries
import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

# Title and description
st.title("URL Text Summarizer")
st.write("""
This app summarizes the text content of any given URL using Hugging Face's pre-trained summarization model.
""")

# Input: URL
url = st.text_input("Enter the URL:", placeholder="https://example.com")

# Summarization function
def summarize_url(url):
    try:
        # Fetch the webpage content
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text content from paragraphs
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        
        # Check if content exists
        if not content.strip():
            return "No readable content found on the page!"
        
        # Initialize the Hugging Face summarization pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Summarize the content
        summary = summarizer(content, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching the URL: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

# Button to trigger summarization
if st.button("Summarize"):
    if url:
        st.write("Fetching and summarizing content...")
        summary = summarize_url(url)
        st.write("### Summary:")
        st.write(summary)
    else:
        st.error("Please enter a valid URL.")
