from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings, create_pinecone_vector_store
from src.constants import index_name, namespace
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
#from langchain.llms import CTransformers
from langchain_community.llms import CTransformers
from dotenv import load_dotenv
from src.prompt import prompt_template
import os
import streamlit as st
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from accelerate import Accelerator
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

embeddings = download_huggingface_embeddings()

vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings, namespace=namespace)

PROMPT = PromptTemplate(template=prompt_template, input_variables=['context','question'])
chain_type_kwargs = {'prompt': PROMPT}

accelerator = Accelerator()

config = {
    'max_new_tokens':256,
    'temperature':0.8,
    'gpu_layers':128
}

llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q4_0.bin',
                    model_type='llama',
                    config=config,
                    gpu_layers=128,
                    callbacks=[StreamingStdOutCallbackHandler()]
                    )

llm, config = accelerator.prepare(llm, config)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)


# @app.route("/")
# def index():
#     return render_template('chat.html')


# if __name__ == '__main__':
#     app.run(debug=True)


st.title('Medical chatbot')

#Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Accept user input
if prompt := st.chat_input('What is up?'):
    # Add user message in chat
    st.session_state.messages.append({'role':'user', 'content': prompt})
    # Display chat messages in chat window
    with st.chat_message('user'):
        st.markdown(prompt)

    # Display assistant responses in chat window
    with st.chat_message('assistant'):
        result = qa.invoke(prompt)
        #print(result)
        response = result['result']
        #response=st.write_stream(qa.invoke(prompt))
        #print(response)
        st.markdown(response)
    
    st.session_state.messages.append({'role': 'assistant', 'content': response})      
    print('Answer provided')
