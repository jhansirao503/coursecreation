from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from llama_index.core import Settings
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from fastapi.responses import StreamingResponse
import os
from langchain_mistralai import MistralAIEmbeddings
from typing import Generator

import time
from langchain_mistralai import ChatMistralAI
import getpass

import argparse
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from gtts import gTTS
from llama_index.llms.mistralai import MistralAI

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))
from constants import CHROMA_SETTINGS

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    persist_directory:str
    person_havenot_dir: str
    topic:str
    hide_source: bool = False
    mute_stream: bool = False
    
os.environ["MISTRAL_API_KEY"]='7Fdsnjymji7prQcv8ylPf8OubFrCbFAf'
@app.post("/query")
def handle_query(request: QueryRequest):
    try:
        # embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key='7Fdsnjymji7prQcv8ylPf8OubFrCbFAf')
        embed_model = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        if request.persist_directory!='':
            persist_directory1 = os.environ.get('PERSIST_DIRECTORY', 'document_db')
            persist_directory=os.path.join(persist_directory1,request.persist_directory)
        else:
            persist_directory=os.environ.get('PERSIST_DIRECTORY',request.person_havenot_dir)
        db = Chroma(persist_directory=persist_directory, embedding_function=embed_model)
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
        callbacks = [] if request.mute_stream else [StreamingStdOutCallbackHandler()]
        llm = ChatMistralAI(
                model="open-mixtral-8x22b",
                temperature=0,
                max_retries=2,
                callbacks=callbacks,
                streaming = True
                # other params...
            )
        
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not request.hide_source
        )

        start = time.time()

        Course_Description='''Assume you are creating a self-learning course; 
        provide a course outline with 3 modules. Each module should have a meaningful title and needs 
        to have a minimum of 3 lessons with a major concept to cover.  
        Under each lesson, identify all the major topics which includes '''+request.topic+''' and print topic title.
        Under each topic, print comprehensive list of Sub-Topics with title  under each topic.


'''
       
        res = qa(Course_Description)
        answer, docs = res['result'], [] if request.hide_source else res['source_documents']
        end = time.time()
        
       


        response = {
            
            "answer": answer,
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true', help='Use this flag to disable printing of source documents used for answers.')
    parser.add_argument("--mute-stream", "-M", action='store_true', help='Use this flag to disable the streaming StdOut callback for LLMs.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()