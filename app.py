import streamlit as st
import PyPDF2
import toml
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import os
from huggingface_hub import hf_hub_download
import huggingface_hub

# Patch to add backward compatibility for cached_download
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = hf_hub_download


class ShireAIAssistant:
    def __init__(self):
        # Load configuration
        try:
            config = toml.load(".streamlit/config.toml")
            self.style_guide_path = config["data"]["style_guide_path"]
        except Exception as e:
            raise RuntimeError(f"Error loading configuration file: {e}")

        # Initialize the AI model
        try:
            self.model = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                torch_dtype=torch.float32,
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing AI model: {e}")

        # Initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing embeddings: {e}")

        self.vector_store = None

    def load_style_guide(self):
        """Load and process the PDF style guide"""
        try:
            text = ""
            with open(self.style_guide_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            return text_splitter.split_text(text)
        except Exception as e:
            raise RuntimeError(f"Error loading style guide: {e}")

    def initialize_knowledge_base(self):
        """Create vector store from style guide"""
        try:
            chunks = self.load_style_guide()

            # Create vector store
            self.vector_store = Chroma.from_texts(
                texts=chunks,
                embedding=self.embeddings
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing knowledge base: {e}")

    def get_relevant_guidelines(self, text: str, k: int = 3):
        """Retrieve relevant style guide sections"""
        if not self.vector_store:
            return "Style guide not loaded. Please initialize the knowledge base first."

        try:
            docs = self.vector_store.similarity_search(text, k=k)
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            raise RuntimeError(f"Error retrieving relevant guidelines: {e}")

    def analyze_text(self, text: str) -> dict:
        """AI analysis of text against style guide"""
        try:
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
            - The rewritten text following the style guide"""

            response = self.model(
                prompt,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7
            )[0]['generated_text']

            return self.parse_response(response)
        except Exception as e:
            raise RuntimeError(f"Error analyzing text: {e}")

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

    if 'assistant' not in st.session_state:
        with st.spinner("Initializing AI assistant..."):
            try:
                st.session_state.assistant = ShireAIAssistant()
                st.session_state.assistant.initialize_knowledge_base()
                st.success("Style guide loaded successfully!")
            except Exception as e:
                st.error(f"Error initializing AI assistant: {str(e)}")

    text = st.text_area(
        "Enter your text:",
        height=200,
        placeholder="Paste your text here for AI-powered style guide review..."
    )

    if text:
        if st.button("Analyze Text"):
            with st.spinner("Analyzing with AI..."):
                try:
                    analysis = st.session_state.assistant.analyze_text(text)
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
                except Exception as e:
                    st.error(f"Error analyzing text: {str(e)}")


if __name__ == "__main__":
    main()
