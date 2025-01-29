from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Pinecone as LangChainPinecone
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

class ChatBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Path to PDF
        pdf_path = "Drugs.pdf"
        loader = PyMuPDFLoader(pdf_path)

        # Load the documents from the PDF
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s) from the PDF.")

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings()

        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index_name = "langchain-demo"

        # Check if the index exists
        if index_name in pc.list_indexes().names():
            self.docsearch = LangChainPinecone.from_existing_index(index_name, self.embeddings)
        else:
            print(f"Creating new index '{index_name}'.")
            pc.create_index(
                name=index_name,
                dimension=768,  # Ensure this matches the embedding model's output dimension
                metric='cosine',
                spec=ServerlessSpec(cloud="aws", region="us-east1")  # Use a supported region
            )
            # Index documents in Pinecone
            self.docsearch = LangChainPinecone.from_documents(docs, self.embeddings, index_name=index_name)

        # Load model using HuggingFaceHub integration in LangChain
        repo_id = "bartowski/Llama-3.2-3B-Instruct-GGUF"
        self.llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.8, "top_k": 50},
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        # Define the prompt template
        template = """
                Question: {question}\n
                Answer: """
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # Set up RAG chain
        retriever = self.docsearch.as_retriever()
        self.rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, user_query):
        # Use the RAG chain to generate a response
        return self.rag_chain.invoke({"question": user_query})

# Instantiate and use the chatbot
if __name__ == "__main__":
    bot = ChatBot()
    user_input = input("Ask me anything: ")
    result = bot.ask(user_input)
    print(result)
