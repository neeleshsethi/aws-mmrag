import boto3
import json
import logging
import os
import time
from utils import utility


from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection

## Instantiate Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

## Creating session with AWS profile
session = boto3.Session(profile_name="bedrock-profile")
credentials = session.get_credentials()

## Bedrock Agent Client
bedrock_agent_client = session.client("bedrock-agent")
## Bedrock Agent Runtime Client
bedrock_agent_runtime_client = session.client("bedrock-agent-runtime")

SERVICE_NAME = "aoss"
DATA_DIR = "../data"
bucket_name = 'test'


class KnowledgeBaseOperations:
    config_filename = '.aoss_config.txt'

    def create_vector_index(self, collection_name, vector_index_name):
        auth = AWSV4SignerAuth(credentials, session.region_name, SERVICE_NAME)
        ## Creating OpenSearch Client
        oss_client = OpenSearch(
            hosts=[
                {
                    "host": f"{collection_name}.{session.region_name}.aoss.amazonaws.com",
                    "port": 443,
                }
            ],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300,
            pool_maxsize=20,
        )
          ## Prepare Index Body
        index_body = dict(
            settings=dict(index=dict(knn=True)),
            mappings=dict(
                properties=dict(
                    vector=dict(type="knn_vector", dimension=1536),
                    text=dict(type="text"),
                    text_metadata=dict(type="text"),
                )
            ),
        )
        ## Create Index
        response = oss_client.indices.create(
            index=vector_index_name, body=json.dumps(index_body)
        )

        logger.info(f"Index Creation Response: {response}")




    def create_knowledge_base(
        self,
        name,
        bedrock_execution_role,
        embedding_model_name,
        oss_collection_name,
        vector_index_name,
    ):
        """
        Method to create Knowledge Base
        """
        ## Create Bedrock Agent Client

        sts_client = session.client("sts")
        aws_account = sts_client.get_caller_identity().get("Account")

        kb = bedrock_agent_client.create_knowledge_base(
            name=name,
            description=f"Knowledge Base: {name}",
            roleArn=f"arn:aws:iam::{aws_account}:role/{bedrock_execution_role}",
            knowledgeBaseConfiguration=dict(
                type="VECTOR",
                vectorKnowledgeBaseConfiguration=dict(
                    embeddingModelArn=f"arn:aws:bedrock:{session.region_name}::foundation-model/{embedding_model_name}"
                ),
            ),
            storageConfiguration=dict(
                type="OPENSEARCH_SERVERLESS",
                opensearchServerlessConfiguration=dict(
                    collectionArn=f"arn:aws:aoss:{session.region_name}:{aws_account}:collection/{oss_collection_name}",
                    vectorIndexName=vector_index_name,
                    fieldMapping=dict(
                        vectorField="vector",
                        textField="text",
                        metadataField="text_metadata",
                    ),
                ),
            ),
        )
        logger.info(f"Knowledge Base Creation Response: {kb}")

    def create_kb_datasource(self, name, kb_Id, bucket_name):
        """
        Method to create Datasource in Knowledge Base
        """
        # Create a DataSource in KnowledgeBase
        kb_datasource = bedrock_agent_client.create_data_source(
            name=name,
            description=f"KB Datasource: {name}",
            knowledgeBaseId=kb_Id,
            dataSourceConfiguration=dict(
                type="S3",
                s3Configuration=dict(
                    bucketArn=f"arn:aws:s3:::{bucket_name}",
                ),
            ),
            vectorIngestionConfiguration=dict(
                chunkingConfiguration=dict(
                    chunkingStrategy="FIXED_SIZE",
                    fixedSizeChunkingConfiguration=dict(
                        maxTokens=512, overlapPercentage=20
                    ),
                )
            ),
        )
        logger.info(f"KB Datasource Creation Response: {kb_datasource}")


    def execute_ingestion_job(self, kb_id, kb_ds_id):
        """
        Method to fetch the documents from S3 and ingest them in data source.
        It does the following during ingestion:
        1. Pre-process the document to extract text
        2. Chunk the extracted text based on the configured chunking size
        3. Create embeddings of each chunk
        4. Write the embedding vectors to the vector database.
        """
        response = bedrock_agent_client.start_ingestion_job(
            knowledgeBaseId=kb_id, dataSourceId=kb_ds_id
        )

        logger.info(f"Job Invocation Response: {response}")

        job = response["ingestionJob"]
        job_id = job["ingestionJobId"]
        # Get job
        while job["status"] != "COMPLETE":
            job_response = bedrock_agent_client.get_ingestion_job(
                knowledgeBaseId=kb_id, dataSourceId=kb_ds_id, ingestionJobId=job_id
            )
            job = job_response["ingestionJob"]
            logger.info(job)
            time.sleep(40)


    def search_using_kb_with_retrieve_and_generate(self, model_id, kb_id, search_text):
        """
        Method to use RetrieveAndGenerate API to convert queries into embeddings, searches the knowledge base,
        and augments the foundation model prompt with the search results as context information
        and then returns the final FM-generated response.
        """
        response = bedrock_agent_runtime_client.retrieve_and_generate(
            input=dict(text=search_text),
            retrieveAndGenerateConfiguration=dict(
                type="KNOWLEDGE_BASE",
                knowledgeBaseConfiguration=dict(
                    knowledgeBaseId=kb_id,
                    modelArn=f"arn:aws:bedrock:{session.region_name}::foundation-model/{model_id}",
                ),
            ),
        )
        ## Log Response
        logger.info(f"Complete Response: {response}")

        ## Retrieve Context
        citations = response["citations"]
        contexts = []
        for citation in citations:
            retrievedReferences = citation["retrievedReferences"]
            for reference in retrievedReferences:
                contexts.append(reference["content"]["text"])

        logger.info(f"Context: {contexts}")

        ## retrieve Response
        search_response = response["output"]["text"]
        logger.info(f"Search Text Response: {search_response}")

    def search_using_kb_with_retrieve(self, kb_id, search_text):
        """
        Method to use Retrieve API to convert user queries into embeddings, searches the knowledge base,
        and returns the relevant results.
        """

        response = bedrock_agent_runtime_client.retrieve(
            retrievalQuery=dict(text=search_text),
            knowledgeBaseId=kb_id,
            retrievalConfiguration=dict(
                vectorSearchConfiguration=dict(numberOfResults=3)
            ),
        )

        ## Log Response
        logger.info(f"Complete Response: {response}")

        ## retrieve Response
        search_response = response["retrievalResults"]
        logger.info(f"Search Text Response: {search_response}")

    def list_knowledge_bases(self):
        """
        Method to fetch the list of knowledge Base
        """

        get_kb_response = bedrock_agent_client.list_knowledge_bases()
        print(get_kb_response)

    def list_kb_datasources(self, kb_id):
        """
        Method to fetch the datasources created with Knowledge base
        """
        get_kb_response = bedrock_agent_client.list_data_sources(knowledgeBaseId=kb_id)
        print(get_kb_response)

    def cleanup(self, collection_name, kb_id, kb_ds_id, vector_index_name):
        """
        Method to cleanup KB datasource, Knowledge base and Vector Index
        """
        auth = AWSV4SignerAuth(credentials, session.region_name, SERVICE_NAME)

        ## Creating OpenSearch Client
        oss_client = OpenSearch(
            hosts=[
                {
                    "host": f"{collection_name}.{session.region_name}.aoss.amazonaws.com",
                    "port": 443,
                }
            ],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300,
            pool_maxsize=20,
        )

        try:
            bedrock_agent_client.delete_data_source(
                dataSourceId=kb_ds_id, knowledgeBaseId=kb_id
            )
            logger.info(f"Datasource {kb_ds_id} is deleted.")
        except Exception as ex:
            logger.error(ex)

        try:
            bedrock_agent_client.delete_knowledge_base(knowledgeBaseId=kb_id)
        except Exception as ex:
            logger.info(f"Knowledge base {kb_id} is deleted.")
            logger.error(ex)

        try:
            oss_client.indices.delete(index=vector_index_name)
            logger.info(f"Vector Index {vector_index_name} is deleted.")
        except Exception as ex:
            logger.error(ex)
            
            
    def upload_document(self):
        s3_client = session.client("s3")
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                s3_client.upload_file(os.path.join(root, file), bucket_name, file)
                logger.info(
                    f"Data gets uploaded for the file: {os.path.join(root, file)}"
                )
        logger.info("Data Upload operation is done.")

    def bedrock_kb_injection(path):
        kb_id = utility.read_key_value(utility.config_filename, 'KB_id')
        ds_id = utility.read_key_value(utility.config_filename, 'DS_id')
        region_name = utility.read_key_value(utility.config_filename, 'Region')
        bucket_name = utility.read_key_value(utility.config_filename, 'S3_bucket_name')
        # Bedrock KB syncs with designated S3 bucket so be careful
        utility.empty_versioned_s3_bucket(bucket_name)
        utility.uploadDirectory(path,bucket_name)
        #ds = bedrock_agent_client.get_data_source(knowledgeBaseId = kb_id, dataSourceId = ds_id)
        start_job_response = bedrock_agent_client.start_ingestion_job(knowledgeBaseId = kb_id, dataSourceId = ds_id)
        job = start_job_response["ingestionJob"]
        while(job['status']!='COMPLETE' ):
            get_job_response = bedrock_agent_client.get_ingestion_job(
            knowledgeBaseId = kb_id,
                dataSourceId = ds_id,
                ingestionJobId = job["ingestionJobId"]
        )
            job = get_job_response["ingestionJob"]
        utility.interactive_sleep(40)
        return job['statistics'], job['status'], kb_id
        
    