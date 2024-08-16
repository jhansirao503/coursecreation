from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import asyncio
import base64
import requests
from fastapi import FastAPI, HTTPException
from playwright.async_api import async_playwright
import time
from io import BytesIO
import asyncio
import json
import base64
import io
from PIL import Image

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

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_mistralai import ChatMistralAI
from constants import CHROMA_SETTINGS

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    persist_directory: str
    person_havenot_dir: str
    hide_source: bool = False
    mute_stream: bool = False
    topic:str
    image:bool = True
    topic_prompt: str

os.environ["MISTRAL_API_KEY"] = 'Dtb6HgWluynve8WqUWbwxaRkH58ko6h5'

async def process_query(request: QueryRequest):
    try:
        embed_model = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        if request.persist_directory:
            persist_directory1 = os.environ.get('PERSIST_DIRECTORY', 'document_db')
            persist_directory = os.path.join(persist_directory1, request.persist_directory)
        else:
            persist_directory = os.environ.get('PERSIST_DIRECTORY', request.person_havenot_dir)
        
        db = Chroma(persist_directory=persist_directory, embedding_function=embed_model)
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
        callbacks = [] if request.mute_stream else [StreamingStdOutCallbackHandler()]
        
        llm = ChatMistralAI(
            model="open-mixtral-8x7b",
            temperature=0.2,
            max_retries=2,
            # other params...
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not request.hide_source
        )

        Course_Description = request.topic_prompt+'''  
                elaborate each topic above in 30 lines with bullet points'''
       
        res = await qa.acall(Course_Description)  # Use the async call method
        answer = res['result'] 
        if request.image==True:
            try:
                search_term=request.topic
                
                base64_image = await generate_image(search_term)
                
                # return {"image": base64_image}
            except Exception as e:
                    base64_image='null' 
            response = {
                'topic':request.topic,
                "answer": answer,
                "image_base64": base64_image
            }

            return response
        else:
            response = {
                'topic':request.topic,
                "answer": answer,
                
            }
            return response
        # try:
        #     search_term=request.topic.split()[-1]
        #     image_url = await fetch_image_url(search_term)
        #     image_data = requests.get(image_url).content
        #     base64_image = base64.b64encode(image_data).decode('utf-8')
            # return {"image_base64": base64_image}
        # except Exception as e:
        #         base64_image='null'                                                                                                              
        
        # response = {
        #     "topic":request.topic,
        #     "answer": answer,
        #     "image_base64": base64_image
        # }

        # return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# async def fetch_image_url(search_term):
#     async with async_playwright() as p:
#         browser = await p.chromium.launch(headless=False)  # Adjust headless as needed
#         page = await browser.new_page()

#         search_url = f"https://www.pexels.com/search/{search_term}"
#         await page.goto(search_url)
#         await page.wait_for_selector("article a img",timeout=5000)

#         image_srcs = await page.evaluate('''() => {
#             return Array.from(document.querySelectorAll('article a img'))
#                 .slice(0, 1)  // Limit to the first image
#                 .map(img => img.src);
#         }''')

#         await browser.close()

#         if not image_srcs:
#             raise HTTPException(status_code=404, detail="Image not found")
        
#         return image_srcs[0]

async def generate_image(prompt):
    url = "https://api.freepik.com/v1/ai/text-to-image"
    
    payload = json.dumps({
      "prompt": prompt
    })
    
    headers = {
      'Content-Type': 'application/json',
      'x-freepik-api-key': 'FPSX0ba7cbfd83f648cab1354834d3e09f8d',
      'Accept': 'application/json'
    }
    
    response = requests.post(url, headers=headers, data=payload)
    
    try:
        response_text = response.text
        a = response_text.split('"')
        base64string = a[5]
        decoded_bytes = base64.b64decode(base64string)

        # Re-encode the image to base64
        buffered = BytesIO(decoded_bytes)
        image = Image.open(buffered)

        # Re-encode image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Return the base64-encoded image in the response
        # return {"image": base64_image}
        # print(base64_image)
        return base64_image
    except:
        print(f"Request failed with status: {response.status}")
        return None




@app.post("/query")
async def handle_query(requests: list[QueryRequest]):
    try:
        tasks = [process_query(request) for request in requests]
        # tasks=process_query(requests)
        results = await asyncio.gather(*tasks)
        return results
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
