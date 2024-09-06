from langchain_community.utilities import GoogleSearchAPIWrapper
import string
import requests
import sys
from bs4 import BeautifulSoup
from botocore.exceptions import ClientError
import boto3
import re, os
from langchain_core.prompts import PromptTemplate
from langchain_core.documents.base import Document
from langchain.prompts.chat import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import FlashrankRerank

from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser





def extract_keywords(input_string):
    # Remove punctuation and convert to lowercase
    input_string = input_string.translate(str.maketrans('', '', string.punctuation)).lower()
    # Split the string into words
    words = input_string.split()
 
    # Define a regular expression pattern to match web searchable keywords
    pattern = r'^[a-zA-Z0-9]+$'
    # Filter out non-keyword words
    keywords = [word for word in words if re.match(pattern, word)]
    # Join the keywords with '+'
    output_string = '+'.join(keywords)
    return re.sub(r'[.-:/"\']', ' ', output_string)

def google_search(query: str, num_results: int=5):
    gsearch = GoogleSearchAPIWrapper(google_api_key=os.getenv("google_api_key"), google_cse_id=os.getenv("google_cse_id"))
    params = {
        "num_results": num_results,  # Number of results to return
        #"exclude": "youtube.com"  # Exclude results from YouTube
    }
    #key_words = re.sub(r'[^a-zA-Z0-9\s]', ' ', extract_keywords(query))
    google_results = gsearch.results(extract_keywords(query), **params)
    #google_results = gsearch.results(query, num_results=num_results)
    documents = []
    urls = []
    for item in google_results:
        try:
            # Send a GET request to the URL
            if ('link' not in item) or('youtube' in item['link'].lower()):
                continue
            response = requests.get(item['link'])
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract the text content from the HTML
            content = soup.get_text()
            if "404 Not Found" not in content:
                # Create a LangChain document
                doc = Document(page_content=content, metadata={'title': item['title'],'source': item['link']})
                documents.append(doc)
                urls. append(item['link'])
    
        except requests.exceptions.RequestException as e:
            print(f"Error parsing URL: {e}")
            pass

    return documents, urls

def retrieval_faiss(query, documents, model_id, embedding_model_id:str, chunk_size:int=6000, over_lap:int=600, max_tokens: int=2048, temperature: int=0.01, top_p: float=0.90, top_k: int=25, doc_num: int=3):
    #text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=over_lap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=over_lap, length_function=len, is_separator_regex=False,)
    docs = text_splitter.split_documents(documents)
    
    # Prepare embedding function
    chat, embedding = config_bedrock(embedding_model_id, model_id, max_tokens, temperature, top_p, top_k)
    
    # Try to get vectordb with FAISS
    db = FAISS.from_documents(docs, embedding)
    retriever = db.as_retriever(search_kwargs={"k": doc_num})


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    messages = [
        ("system", """Your are a helpful assistant to provide comprehensive and truthful answers to questions, \n
                    drawing upon all relevant information contained within the specified in {context}. \n 
                    You add value by analyzing the situation and offering insights to enrich your answer. \n
                    Simply say I don't know if you can not find any evidence to match the question. \n
                    """),
        #MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)

    # Reranker
    compression_retriever = ContextualCompressionRetriever(
        base_compressor= FlashrankRerank(), base_retriever=retriever
    )

    rag_chain = (
        #{"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        #| RunnableParallel(answer=hub.pull("rlm/rag-prompt") | chat |format_docs, question=itemgetter("question") ) 
        RunnableParallel(context=compression_retriever | format_docs, question=RunnablePassthrough() )
        | prompt_template
        | chat
        | StrOutputParser()
    )

    results = rag_chain.invoke(query)
    return results

def config_bedrock(embedding_model_id, model_id, max_tokens, temperature, top_p, top_k):
    bedrock_client = boto3.client('bedrock-runtime')
    embedding_bedrock = BedrockEmbeddings(client=bedrock_client, model_id=embedding_model_id)
    model_kwargs =  { 
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        #"stop_sequences": ["\n\nHuman"],
    }
    chat = BedrockChat(
        model_id=model_id, client=bedrock_client, model_kwargs=model_kwargs
    )
    #llm = Bedrock(
    #    model_id=model_id, client=bedrock_client, model_kwargs=model_kwargs
    #)

    return chat, embedding_bedrock

def classify_query(query, classes: str, modelId: str):
    """
    Classify a query into 'Tech', 'Health', or 'General' using an LLM.

    :param query: The query string to classify.
    :param openai_api_key: Your OpenAI API key.
    :return: A string classification: 'Tech', 'Health', or 'General'.
    """
    bedrock_client = boto3.client('bedrock-runtime')
    
    # Constructing the prompt for the LLM
    prompt = f"Human:Classify the following query into one of these categories: {classes}.\n\nQuery: {query}\n\n Please answer directly with the catergory name only. \n\n  AI:"
    payload = {
            "modelId": modelId,
            "contentType": "application/json",
            "accept": "application/json",
            "body": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": 0.01,
                "top_k": 250,
                "top_p": 0.95,
                #"stop_sequences": stop_sequence,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ]
                    }
                ]
            }
        }
    try:
        # Convert the payload to bytes
        body_bytes = json.dumps(payload['body']).encode('utf-8')
        # Invoke the model
        response = bedrock_client.invoke_model(
            body=body_bytes,
            contentType=payload['contentType'],
            accept=payload['accept'],
            modelId=payload['modelId']
        )


        #response = bedrock_client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
        response_body = json.loads(response.get('body').read())
        classification = ''.join([item['text'] for item in response_body['content'] if item.get('type') == 'text'])
        # Assuming the most likely category is returned directly as the output text
        #classification = response.choices[0].text.strip()
        return classification
    except Exception as e:
        print(f"Error classifying query: {e}")
        return "Error"
