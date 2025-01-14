import os
import faiss
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import streamlit as st
import textwrap
import requests
# create env enviroment
# python -m venv gemini python==3.10
# cmd
# paste path of genai\Scripts\activate.bat
# run file : streamlit run app_final_gemini.py

# Load environment variables
load_dotenv()
# FAISSE 
class PDFQuestionAnswerer:
    def __init__(self, api_key):
        """Initialize the question answerer with API keys."""
        self.api_key = api_key  # Store the API key
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use a Hugging Face model for embeddings
        self.vector_store = None

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def process_pdfs(self, pdf_directory):
        """Process multiple PDFs and create a vector store."""
        all_texts = []
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in directory: {pdf_directory}")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            text = self.extract_text_from_pdf(pdf_path)
            if text:
                all_texts.append(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.create_documents(all_texts)

        # Create FAISS index from documents using Hugging Face embeddings
        self.vector_store = FAISS.from_documents(
            chunks,
            self.embeddings
        )

        # Check that vector_store is created successfully
        if self.vector_store is not None:
            # Save the FAISS index to a file using pickle
            with open('faiss_vector_store.pkl', 'wb') as f:
                pickle.dump(self.vector_store, f)
            print(f"FAISS vector store saved successfully.")
        else:
            print("Error: FAISS vector store creation failed.")

    def load_vector_store(self):
        """Load the FAISS vector store from a saved file."""
        try:
            with open('faiss_vector_store.pkl', 'rb') as f:
                self.vector_store = pickle.load(f)
            print("FAISS vector store loaded successfully.")
        except FileNotFoundError:
            print("No saved FAISS vector store found.")
            self.vector_store = None

    def get_relevant_chunks(self, query, k=3):
        """Retrieve the most relevant chunks for a query."""
        if not self.vector_store:
            raise ValueError("No PDFs have been processed yet. Call process_pdfs first.")

        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def format_prompt(self, query, relevant_chunks):
        """Format the prompt for GEMINI with context."""
        context = "\n\n".join(relevant_chunks)
        prompt = f"""\n\nHuman: Here are some relevant passages from the documents:

{context}

Based on the passages above, please answer this question: {query}

If the answer cannot be fully determined from the provided passages, please say so. Include specific references to the source material where possible.

Assistant:"""
        return prompt



    def ask_question(self, query):
        """Ask a question about the processed PDFs using Google's Gemini API."""
        if self.vector_store is None:
            print("Error: No vector store available. Please process PDFs first.")
            return "Error: No PDFs processed yet."

        # Retrieve relevant chunks
        # Retrieve relevant chunks
        relevant_chunks = self.get_relevant_chunks(query)
        if relevant_chunks is None:
          return "Error retrieving relevant context."

        prompt = self.format_prompt(query, relevant_chunks)

        # Define the Gemini API endpoint
        gemini_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

        # Prepare headers and payload
        headers = {
            "Content-Type": "application/json",
             "x-goog-api-key": self.api_key # Using x-goog-api-key instead of Authorization Bearer
        }
        payload = {
            "contents": [
              {
                "parts": [
                  {
                    "text": prompt
                  }
                ]
              }
            ]
          }

        try:
            # Send the request to Gemini
            response = requests.post(gemini_endpoint, headers=headers, json=payload)
            # response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raise an error for HTTP issues

             # Extract the generated response text
            response_data = response.json()

            # Extracting text from the response structure (check if the response is valid)
            if response_data and "candidates" in response_data and response_data["candidates"]:
              # Since we expect a single response, access the first candidate and part
              first_candidate = response_data["candidates"][0]
              if "content" in first_candidate and "parts" in first_candidate["content"] and first_candidate["content"]["parts"]:
                  first_part = first_candidate["content"]["parts"][0]
                  if "text" in first_part:
                      generated_text = first_part["text"]
                      return generated_text
              # Handle case where the text cannot be extracted
              print("Error: Response structure does not match expected format.")
              return "Error: Could not extract response text."
            else:
              print("Error: No candidates in the response.")
              return "Error: Could not generate a response."

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Gemini API: {e}")
            return f"Error: {e}"


        
def main():
    try:
        # Load API keys and configurations
        # claude_api_key = os.getenv("CLAUDE_API_KEY")
        api_key = os.getenv("api_key")
      
        # pdf_dir = os.getenv("PDF_DIR", "./pdfs")

        if not api_key:
            raise EnvironmentError("API key for is not set in the environment variables.")

        # Initialize the system
        qa_system = PDFQuestionAnswerer(api_key)

        # Load the saved vector store, if available
        qa_system.load_vector_store()

        st.title("PDF Question Answering System")
        st.sidebar.header("Upload PDFs")

        uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

        # Handle PDF uploads and processing
        if uploaded_files:
            pdf_directory = "./temp_pdfs"
            os.makedirs(pdf_directory, exist_ok=True)

            for uploaded_file in uploaded_files:
                file_path = os.path.join(pdf_directory, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

            st.sidebar.success("PDFs uploaded successfully.")

            # Process PDFs and store vector store in session if not processed yet
            if st.sidebar.button("Process PDFs") and qa_system.vector_store is None:
                qa_system.process_pdfs(pdf_directory)
                st.sidebar.success("PDFs processed successfully.")
                qa_system.load_vector_store()  # Reload the vector store after processing

        # Check if PDFs have been processed and vector store exists
        if qa_system.vector_store is None:
            st.warning("Please upload and process the PDFs first.")
        else:
            query = st.text_input("Ask a question about the PDFs:")

            if query:
                answer = qa_system.ask_question(query)
                st.write("### Answer:")
                st.write(textwrap.fill(answer, width=80))

    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()