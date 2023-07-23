import asyncio
# from asyncore import loop
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain.document_loaders import UnstructuredPDFLoader, DirectoryLoader, PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
#from langchain import Documents
from typing import Dict
import os
import shutil
import sys
import re

# basic setup
os.environ['OPENAI_API_KEY'] = '[YOUR_OPENAI_API_KEY]'
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
pdf_folder_path = r"[YOUR_FOLDER_PATH]"  
persist_directory = 'db'

# Load documents and create database when application starts
loader = DirectoryLoader(pdf_folder_path, glob="./*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# docs = []
for doc in docs:
    # Use metadata attribute to store the filename
    doc.metadata['filename'] = os.path.basename(doc.metadata['source']).replace('.pdf', '')
    parts = doc.metadata['filename'].split('_')
    if len(parts) == 4:
        metadata = {"analyst": parts[0], "company": parts[1], "year": parts[2], "report_type": parts[3]}
    elif len(parts) >= 5:
        metadata = {"analyst": parts[0], "company": parts[1], "year": parts[2], "quarter": parts[3], "report_type": parts[4]}
    else:
        metadata = {"source": doc.metadata['filename']}
    doc.metadata.update(metadata)


# Splitting the document into text chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
texts = text_splitter.split_documents(docs)

# Create a database
embedding = OpenAIEmbeddings(model = "text-embedding-ada-002")

vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding, 
                                 persist_directory=persist_directory)

# Persist the db to disk
vectordb.persist()
vectordb = None


@app.get("/", response_class=HTMLResponse)
async def root():
    with open('templates/home.html', 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/handle_query")
async def handle_query(query: str = Form(...), docs=docs):
    
    # Now we can load the persisted database from disk, and use it as normal. 
    vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

    # create retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})  # increase number of documents considered

    # convert query to lower case and remove punctuation
    query_filter = re.sub(r'[^\w\s]', '', query.lower())

    # Filter documents based on the query
    keywords = query_filter.lower().split()
    filtered_docs = []
    for doc in docs:
        if any(keyword in doc.metadata.get("analyst", '').lower() for keyword in keywords):
            filtered_docs.append(doc)
    # something filtered by the analyst, use the filtered docs for further filtering
    #if filtered_docs==[]:
        # return {"detail": "No documents found for given analyst"}
    if filtered_docs!=[]:
        docs = filtered_docs
        print("1")

    filtered_docs = []
    for doc in docs:
        if any(keyword==doc.metadata.get("company", '').lower() for keyword in keywords):
            filtered_docs.append(doc)
    if filtered_docs==[]:
        return {"detail": "No documents found for given company"}

    # Split the filtered documents into text chunks
    filtered_texts = text_splitter.split_documents(filtered_docs)

    # Store the 'analyst' and 'company' of each filtered text in a list of tuples
    results = [(text.metadata.get('filename', 'filename not found'), 
                           text.metadata.get('analyst', 'Analyst not found'),
                           text.metadata.get('company', 'Company not found')) for text in filtered_texts]

    # Print the list
    for filename, analyst, company in results:
        print(f"FileName: {filename}, Analyst: {analyst}, Company: {company}")

    # Create a temporary vectorstore for the filtered documents
    filtered_vectordb = Chroma.from_documents(documents=filtered_texts, 
                                              embedding=embedding)

    # Create retriever for the filtered documents
    filtered_retriever = filtered_vectordb.as_retriever(search_kwargs={"k": 5})


    # Set up the turbo LLM
    turbo_llm = ChatOpenAI(
        temperature=0, # adjust temperature for more/less randomness
        model_name='gpt-3.5-turbo-16k-0613'
    )
    
    # create the chain
    qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm, 
                              chain_type="stuff", 
                              retriever=filtered_retriever, 
                              return_source_documents=True)

    # Extract document names from query and add them to the metadata
    document_names = [doc_name for doc_name in query.split() 
                      if any(doc_name in doc.metadata['source'] 
                      for doc in filtered_docs)]
    for doc in docs:
        if doc.metadata['source'] in document_names:
            doc.metadata['query'] = query
    
    guidance = 'use 2000 to 4000 tokens, professional tone'
    
    def run_qa_chain():
        # return qa_chain(query)
        return qa_chain(query + guidance)

    with ThreadPoolExecutor() as executor:
        llm_response = await asyncio.to_thread(run_qa_chain)

    
    return llm_response


@app.post("/query")
async def get_response(body: Dict[str, str]):
    query = body.get("query")
    if query is None:
        return {"detail": "Query not provided"}
    response_0 = await handle_query(query)
    if 'source_documents' in response_0:
        sources_all = [doc.metadata['source'] for doc in response_0["source_documents"]]
    else:
        sources_all = []  # or some other default value that makes sense in your context
    if 'result' in response_0:
        answer = response_0['result']
    else:
        answer = "retry"  # or some other default value that makes sense in your context
    return {"answer": answer,
            "source": sources_all}

