import streamlit as st
import PyPDF2 as pdf
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_cohere import ChatCohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

from langchain_cohere import CohereEmbeddings


# Load environment variables from .env file
load_dotenv()
# load_dotenv(override=True)

# Access API keys from environment variables
# cohere_api_key = "exySgfjmdHGJZb0HiRsawoYHODdEDMbEYY1VKksj"
# pinecone_api_key = "pcsk_5jsRHG_3jftdXNPiSzNYbMaKQK84BFGYU1JaydoD4714SjAqZUHnDpvcPGYUrTrM88K56w"
cohere_api_key = os.getenv("COHERE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
# st.text_area("api", cohere_api_key, height=300)

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

def initialize_pinecone_and_cohere():
    # Initialize Cohere embeddings
    try:
        embeddings = CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model="embed-english-v3.0"
        )
    except Exception as e:
        st.error(f"Failed to initialize Embedded model: {str(e)}")


    # Initialize Pinecone
    try:
        pinecone_client = Pinecone(
            api_key=pinecone_api_key
        )
        return pinecone_client, embeddings
    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {str(e)}")
        return None, embeddings

# Streamlit app setup
st.set_page_config(page_title="Document Q&A with Cohere and Pinecone", page_icon="📄")
st.title("Document Q&A with Cohere and Pinecone")

# Upload document section
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from the document..."):
        uploaded_text = extract_text_from_pdf(uploaded_file)
        st.success("Document text extracted successfully!")
        st.text_area("Extracted Text", uploaded_text, height=300)

    if cohere_api_key and pinecone_api_key:
        try:
            # Initialize services
            pinecone_client, embeddings = initialize_pinecone_and_cohere()

            if not pinecone_client:
                st.error("Failed to initialize Pinecone client")
                st.stop()

            # Create index if it doesn't exist
            index_name = "docqa"
            dimension = 1024  # Dimension for embed-english-v3.0

            # List existing indexes
            existing_indexes = pinecone_client.list_indexes()

            # Check if index exists and create if necessary
            if index_name not in [index.name for index in existing_indexes]:
                try:
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                    pinecone_client.create_index(
                        name=index_name,
                        dimension=dimension,
                        metric='cosine',
                        spec=spec
                    )
                    st.success(f"Created new Pinecone index: {index_name}")
                except Exception as e:
                    st.error(f"Error creating index: {str(e)}")
                    st.stop()

            # Get the index
            index = pinecone_client.Index(index_name)

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )
            texts = text_splitter.split_text(uploaded_text)

            # Create embeddings and upsert to Pinecone
            for i, text_chunk in enumerate(texts):
                with st.spinner(f'Processing chunk {i+1}/{len(texts)}...'):
                    # Create embedding for the chunk
                    embedding = embeddings.embed_query(text_chunk)
                    
                    # Create a unique ID for this chunk
                    chunk_id = str(uuid.uuid4())
                    # Upsert to Pinecone
                    index.upsert(vectors=[{
                        'id': chunk_id,
                        'values': embedding,
                        'metadata': {'text': text_chunk}
                    }])

            st.success(f"Successfully processed {len(texts)} chunks")

            # Question answering section
            question = st.text_input("Enter your question:")
            if question:
                with st.spinner("Processing your question..."):
                    # Create question embedding
                    question_embedding = embeddings.embed_query(question)
                    
                    # Query Pinecone
                    query_response = index.query(
                        vector=question_embedding,
                        top_k=3,
                        include_metadata=True
                    )
                    
                    # Extract relevant contexts
                    contexts = [match.metadata['text'] for match in query_response.matches]
                    context = "\n".join(contexts)
                    
                    # Initialize Cohere for response generation
                    llm = ChatCohere(
                        model="command-r-plus-08-2024",
                        temperature=0.7,
                        cohere_api_key=cohere_api_key
                    )
                    
                    # Generate response
                    prompt = f"""Based on the following context, please answer the question. 
                    If the answer cannot be found in the context, say so.

                    Context:
                    {context}

                    Question: {question}

                    Answer:"""
                    
                    # st.text_area("Prompt: " , prompt, height=300)
                    response = llm.invoke(prompt)
                    st.success("Answer:")
                    st.write(response.content)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please check your API keys and environment settings.")
            st.write("Debug info:", {
                "cohere_key_exists": bool(cohere_api_key),
                "pinecone_key_exists": bool(pinecone_api_key)
            })
    else:
        st.warning("Please ensure all required environment variables are set.")