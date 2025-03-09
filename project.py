import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Function to extract text from pdf
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

#function to rank Resume based on job desc
def rank_resumes(job_description, resumes):
    #combining job desc with resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    #Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return cosine_similarities

#Streamlit app
st.title("AI Resume Screenning and Ranking")

#Job Description input
st.header("Job Description")
job_description = st.text_area("Enter job Description")

#File Uploader
st.header("Upload Resume")
upload_files = st.file_uploader("Upload PDF files", type = ['pdf'], accept_multiple_files=True)

if upload_files and job_description:
    st.header("Ranking Resumes")

    resumes = []
    for file in upload_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    #Rank Resumes
    scores = rank_resumes(job_description,resumes)

    #display scores
    results = pd.DataFrame({"Resume": [file.name for file in upload_files], "Score": scores})
    results = results.sort_values(by = "Score", ascending=False)

    st.write(results)