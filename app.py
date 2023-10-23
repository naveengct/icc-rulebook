import streamlit as st
import time
import datetime
import os
from dotenv import load_dotenv
from datetime import datetime
from llama_index import (GPTVectorStoreIndex, ServiceContext,
                         SimpleDirectoryReader)
from llama_index.vector_stores.qdrant import QdrantVectorStore
import openai

# Load environment variables
dotenv_path = '.env'
load_dotenv(dotenv_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

from llama_index import StorageContext, load_index_from_storage

service_context = ServiceContext.from_defaults(chunk_size=512)
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="embedding")

# load index
loaded_index = load_index_from_storage(storage_context)

loaded_query_engine = loaded_index.as_query_engine(similarity_top_k = 2, streaming=True)

st.header("Crick Bot")
input_data = st.text_area('')

if len(input_data):
    response = loaded_query_engine.query(input_data)


    message_placeholder = st.empty()
    full_response = ""

    # Simulate stream of response with milliseconds delay
    for chunk in response.response_gen:
        full_response += chunk 
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")

    message_placeholder.markdown(full_response)