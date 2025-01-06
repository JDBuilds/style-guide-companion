import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import pipeline
import torch
import os

class ShireAIAssistant:
    def __init__(self):
        # Initialize the AI model
        self.model = pipeline(
            "text-generation",
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Load and process style guide
        self.vector_store = self.initialize_knowledge_base()
    
    def load_style_guide(self):
        """Load and process the PDF style guide"""
        style_guide_path = "data/style_guide.pdf"
        text = ""
        
        with open(style_guide_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        return text_splitter.split_text(text)

    def initialize_knowledge_base(self):
        """Create vector store from style guide"""
        chunks = self.load_style_guide()
        
        # Create vector store
        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings
        )
        
        return vector_store
    
    def get_relevant_guidelines(self, text: str, k: int = 3):
        """Retrieve relevant style guide sections"""
        docs = self.vector_store.similarity_search(text, k=k)
        return "\n".join([doc.page_content for doc in docs])
    
    def analyze_text(self, text: str) -> dict:
        """AI analysis of text against style guide"""
        # Get relevant guidelines
        guidelines = self.get_relevant_guidelines(text)
        
        prompt = f"""You are a writing assistant for the Mornington Peninsula Shire.
        Review and improve the following text according to these relevant style guide sections:

        {guidelines}

        Text to analyze: {text}

        Provide your response in the following format:
        VIOLATIONS:
        - List any style guide violations
        
        SUGGESTIONS:
        - List specific improvements
        
        IMPROVED VERSION:
        - The rewritten text following the style guide

        Be specific and reference the style guide in your suggestions."""

        # Generate AI response
        response = self.model(
            prompt,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True
        )[0]['generated_text']
        
        # Parse response sections
        sections = self.parse_response(response)
        
        return sections
    
    def parse_response(self, response: str) -> dict:
        """Parse the AI response into structured feedback"""
        sections = {
            'violations': [],
            'suggestions': [],
            'improved_text': ""
        }
        
        current_section = None
        for line in response.split('\n'):
            if 'VIOLATIONS:' in line:
                current_section = 'violations'
            elif 'SUGGESTIONS:' in line:
                current_section = 'suggestions'
            elif 'IMPROVED VERSION:' in line:
                current_section = 'improved_text'
            elif line.strip() and current_section:
                if current_section in ['violations', 'suggestions']:
                    if line.strip().startswith('-'):
                        sections[current_section].append(line.strip()[1:].strip())
                else:
                    sections[current_section] += line.strip() + "\n"
        
        return sections

def main():
    st.title("Shire AI Style Assistant")
    st.write("I help ensure your writing follows the Shire's style guide.")
    
    # Initialize assistant
    if 'assistant' not in st.session_state:
        with st.spinner("Initializing AI assistant..."):
            st.session_state.assistant = ShireAIAssistant()
    
    # Text input
    text = st.text_area(
        "Enter your text:",
        height=200,
        placeholder="Paste your text here for AI-powered style guide review..."
    )
    
    if text:
        if st.button("Analyze Text"):
            with st.spinner("Analyzing with AI..."):
                analysis = st.session_state.assistant.analyze_text(text)
                
                # Display results
                if analysis['violations']:
                    st.subheader("Style Guide Violations")
                    for violation in analysis['violations']:
                        st.warning(violation)
                
                if analysis['suggestions']:
                    st.subheader("Improvement Suggestions")
                    for suggestion in analysis['suggestions']:
                        st.info(suggestion)
                
                st.subheader("AI-Improved Version")
                st.write(analysis['improved_text'])

if __name__ == "__main__":
    main()
