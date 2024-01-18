import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import logging
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from streamlit_chat import message
import config
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
 
load_dotenv()

#Creating the chatbot interface
st.title("LLM-Powered Chatbot for Intelligent Conversations")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []



# Define a function to clear the input text
def clear_input_text():
    global input_text
    input_text = ""

# We will get the user's input by calling the get_text function
def get_text():
    global input_text
    input_text = st.text_input("Ask your Question", key="input", on_change=clear_input_text)
    return input_text

# Initialize logging with the specified configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_FILE),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)

# Define answer generation function
def answer(prompt: str, persist_directory: str = config.PERSIST_DIR) -> str:
    
    # Log a message indicating that the function has started
    LOGGER.info(f"Start answering based on prompt: {prompt}.")
    
    # Create a prompt template using a template from the config module and input variables
    # representing the context and question.
    prompt_template = PromptTemplate(template=config.prompt_template, input_variables=["context", "question"])
    
    # Load a QA chain using an OpenAI object, a chain type, and a prompt template.
    doc_chain = load_qa_chain(
        llm=OpenAI(
            
            model_name="gpt-3.5-turbo",
            temperature=0,
            max_tokens=100,
        ),
        chain_type="stuff",
        prompt=prompt_template,
    )
    
    # Log a message indicating the number of chunks to be considered when answering the user's query.
    LOGGER.info(f"The top {config.k} chunks are considered to answer the user's query.")
    
    # Create a VectorDBQA object using a vector store, a QA chain, and a number of chunks to consider.
    qa = VectorDBQA(vectorstore=docsearch, combine_documents_chain=doc_chain, k=config.k)
    
    # Call the VectorDBQA object to generate an answer to the prompt.
    result = qa({"query": prompt})
    answer = result["result"]
    
    # Log a message indicating the answer that was generated
    LOGGER.info(f"The returned answer is: {answer}")
    
    # Log a message indicating that the function has finished and return the answer.
    LOGGER.info(f"Answering module over.")
    return answer


def main():
    # upload a PDF file
    """pdf = st.file_uploader("Upload your PDF", type='pdf')
     # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()"""
       # text_splitter = RecursiveCharacterTextSplitter(
        #    chunk_size=1000,
         #   chunk_overlap=0,
          #  length_function=len)
        # Load documents from the specified directory using a DirectoryLoader object
    loader = DirectoryLoader(config.FILE_DIR, glob='*.pdf')
    documents = loader.load()
    #chunks = text_splitter.split_text(text=text)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # Split the documents into chunks of size 1000 using a CharacterTextSplitter object
    chunks = text_splitter.split_documents(documents)
    # Create a vector store from the chunks using an OpenAIEmbeddings object and a Chroma object
    embeddings = OpenAIEmbeddings()
    global docsearch
    docsearch = Chroma.from_documents(chunks, embeddings,persist_directory='./data')


    user_input = get_text()

    if user_input:
        output = answer(user_input)
        # store the output 
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


# Run the app
if __name__ == "__main__":
    main()

            