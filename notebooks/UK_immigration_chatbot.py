import os
import requests
import logging
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Haystack imports
from haystack import Document, Pipeline
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.base import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter

# For Ollama integration
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaGenerator:
    """Custom Haystack component for Ollama LLM integration"""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
    
    def run(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using Ollama"""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': kwargs.get('temperature', 0.7),
                    'top_p': kwargs.get('top_p', 0.9),
                    'max_tokens': kwargs.get('max_tokens', 512)
                }
            )
            return {"replies": [response['message']['content']]}
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            return {"replies": ["I apologize, but I'm having trouble generating a response right now."]}

class UKImmigrationDataFetcher:
    """Fetches data from UK government APIs and websites"""
    
    def __init__(self):
        self.base_urls = {
            'gov_uk': 'https://www.gov.uk',
            'visa_api': 'https://www.gov.uk/api/content',
        }
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'UK Immigration Chatbot/1.0'
        })
    
    def fetch_immigration_content(self) -> List[Document]:
        """Fetch immigration-related content from UK government sources"""
        documents = []
        
        # Key immigration pages to fetch
        immigration_pages = [
            '/browse/visas-immigration',
            '/browse/visas-immigration/arrive-in-the-uk',
            '/browse/visas-immigration/eu-eea-commonwealth',
            '/browse/visas-immigration/family-visas',
            '/browse/visas-immigration/student-visas',
            '/browse/visas-immigration/work-visas',
            '/check-uk-visa',
            '/apply-uk-visa',
            '/uk-border-control',
            '/settled-status-eu-citizens-families',
            '/tier-2-general',
            '/student-visa',
            '/family-permit-eea',
            '/spouse-visa',
            '/uk-ancestry-visa',
            '/global-talent-visa'
        ]
        
        for page in immigration_pages:
            try:
                url = f"{self.base_urls['visa_api']}{page}"
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    content = response.json()
                    
                    # Extract relevant content
                    title = content.get('title', '')
                    description = content.get('description', '')
                    details = content.get('details', {})
                    body = details.get('body', '') if isinstance(details, dict) else str(details)
                    
                    if body or description:
                        doc_content = f"Title: {title}\n\nDescription: {description}\n\nContent: {body}"
                        
                        document = Document(
                            content=doc_content,
                            meta={
                                'title': title,
                                'url': f"{self.base_urls['gov_uk']}{page}",
                                'source': 'UK Government',
                                'category': 'Immigration',
                                'last_updated': datetime.now().isoformat()
                            }
                        )
                        documents.append(document)
                        logger.info(f"Fetched content for: {title}")
                
            except Exception as e:
                logger.error(f"Error fetching {page}: {e}")
                continue
        
        # Add some static immigration information as fallback
        if not documents:
            documents = self._get_static_immigration_content()
        
        return documents
    
    def _get_static_immigration_content(self) -> List[Document]:
        """Fallback static immigration content"""
        static_content = [
            {
                'title': 'UK Visa Types Overview',
                'content': '''The UK offers various visa types including:
                
                Work Visas:
                - Skilled Worker visa (formerly Tier 2 General)
                - Global Talent visa
                - Start-up and Innovator visas
                - Intra-company Transfer visa
                
                Study Visas:
                - Student visa (formerly Tier 4)
                - Child Student visa
                
                Family Visas:
                - Spouse/Partner visa
                - Fiancé(e) visa
                - Parent visa
                - Child visa
                
                Visit Visas:
                - Standard Visitor visa
                - Business Visitor visa
                - Transit visa
                
                Settlement:
                - Indefinite Leave to Remain (ILR)
                - British Citizenship
                - EU Settlement Scheme''',
                'category': 'Visa Types'
            },
            {
                'title': 'Visa Application Process',
                'content': '''General UK visa application process:
                
                1. Check if you need a visa
                2. Choose the correct visa type
                3. Complete online application
                4. Pay visa fee and healthcare surcharge
                5. Book biometric appointment
                6. Submit documents
                7. Attend interview (if required)
                8. Wait for decision
                
                Processing times vary by visa type:
                - Visit visas: 3 weeks
                - Work visas: 3-8 weeks
                - Family visas: 2-12 weeks
                - Settlement applications: 6 months''',
                'category': 'Application Process'
            },
            {
                'title': 'Points-Based Immigration System',
                'content': '''The UK uses a points-based immigration system:
                
                Skilled Worker visa requirements:
                - Job offer from approved sponsor (20 points)
                - Job at appropriate skill level (20 points)
                - English language proficiency (10 points)
                - Salary threshold met (20+ points)
                
                Minimum salary: £25,600 or going rate for job
                Higher salaries can earn more points
                
                Additional points available for:
                - PhD in relevant subject
                - Job in shortage occupation
                - Healthcare/education roles''',
                'category': 'Points System'
            }
        ]
        
        documents = []
        for content in static_content:
            document = Document(
                content=f"Title: {content['title']}\n\nContent: {content['content']}",
                meta={
                    'title': content['title'],
                    'source': 'Static Content',
                    'category': content['category'],
                    'last_updated': datetime.now().isoformat()
                }
            )
            documents.append(document)
        
        return documents

class UKImmigrationChatbot:
    """Main chatbot class with Haystack RAG pipeline"""
    
    def __init__(self, ollama_model: str = "llama2"):
        self.ollama_model = ollama_model
        self.document_store = InMemoryDocumentStore()
        self.data_fetcher = UKImmigrationDataFetcher()
        
        # Initialize components
        self.embedder = SentenceTransformersTextEmbedder(model="all-MiniLM-L6-v2")
        self.doc_embedder = SentenceTransformersDocumentEmbedder(model="all-MiniLM-L6-v2")
        self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store)
        self.generator = OllamaGenerator(model=ollama_model)
        
        # Document processing components
        self.cleaner = DocumentCleaner()
        self.splitter = DocumentSplitter(split_by="word", split_length=200, split_overlap=50)
        self.writer = DocumentWriter(document_store=self.document_store)
        
        # Build pipelines
        self._build_indexing_pipeline()
        self._build_query_pipeline()
        
        # Load initial data
        self._initialize_knowledge_base()
    
    def _build_indexing_pipeline(self):
        """Build document indexing pipeline"""
        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("cleaner", self.cleaner)
        self.indexing_pipeline.add_component("splitter", self.splitter)
        self.indexing_pipeline.add_component("embedder", self.doc_embedder)
        self.indexing_pipeline.add_component("writer", self.writer)
        
        # Connect components
        self.indexing_pipeline.connect("cleaner", "splitter")
        self.indexing_pipeline.connect("splitter", "embedder")
        self.indexing_pipeline.connect("embedder", "writer")
    
    def _build_query_pipeline(self):
        """Build RAG query pipeline"""
        prompt_template = """
        You are a helpful UK immigration assistant. Use the following context to answer the user's question about UK immigration, visas, and related topics.
        
        Context:
        {% for doc in documents %}
        {{ doc.content }}
        ---
        {% endfor %}
        
        Question: {{ question }}
        
        Instructions:
        - Provide accurate, helpful information based on the context
        - If you're unsure about something, say so
        - Include relevant visa types, requirements, or processes
        - Mention if the user should check the official gov.uk website for the most current information
        - Be concise but comprehensive
        
        Answer:
        """
        
        prompt_builder = PromptBuilder(template=prompt_template)
        
        self.query_pipeline = Pipeline()
        self.query_pipeline.add_component("text_embedder", self.embedder)
        self.query_pipeline.add_component("retriever", self.retriever)
        self.query_pipeline.add_component("prompt_builder", prompt_builder)
        self.query_pipeline.add_component("generator", self.generator)
        
        # Connect components
        self.query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self.query_pipeline.connect("retriever", "prompt_builder.documents")
        self.query_pipeline.connect("prompt_builder", "generator.prompt")
    
    def _initialize_knowledge_base(self):
        """Load immigration data into the knowledge base"""
        logger.info("Fetching UK immigration data...")
        documents = self.data_fetcher.fetch_immigration_content()
        
        if documents:
            logger.info(f"Indexing {len(documents)} documents...")
            self.indexing_pipeline.run({"cleaner": {"documents": documents}})
            logger.info("Knowledge base initialized successfully")
        else:
            logger.warning("No documents fetched - using minimal knowledge base")
    
    def update_knowledge_base(self):
        """Refresh the knowledge base with latest data"""
        logger.info("Updating knowledge base...")
        self.document_store.delete_documents()
        self._initialize_knowledge_base()
    
    def chat(self, question: str, top_k: int = 5) -> str:
        """Process user question and return response"""
        try:
            result = self.query_pipeline.run({
                "text_embedder": {"text": question},
                "retriever": {"top_k": top_k},
                "prompt_builder": {"question": question}
            })
            
            return result["generator"]["replies"][0]
        
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return "I apologize, but I encountered an error while processing your question. Please try rephrasing it or check if the system is running properly."
    
    def get_relevant_documents(self, question: str, top_k: int = 3) -> List[Document]:
        """Get relevant documents for a question (for debugging)"""
        embedding = self.embedder.run(text=question)
        result = self.retriever.run(query_embedding=embedding["embedding"], top_k=top_k)
        return result["documents"]

def main():
    """Main function to run the chatbot"""
    print("🇬🇧 UK Immigration Chatbot")
    print("=" * 40)
    print("Initializing chatbot with Haystack RAG and Ollama...")
    
    try:
        # Initialize chatbot (you can change the model)
        chatbot = UKImmigrationChatbot(ollama_model="mistral:instruct")  # or "mistral", "codellama", etc.
        
        print("\nChatbot ready! Ask me anything about UK immigration.")
        print("Type 'quit' to exit, 'update' to refresh knowledge base, or 'debug <question>' to see retrieved documents.")
        print("-" * 40)
        
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye! 👋")
                break
            elif user_input.lower() == 'update':
                chatbot.update_knowledge_base()
                print("✅ Knowledge base updated!")
                continue
            elif user_input.lower().startswith('debug '):
                question = user_input[6:]
                docs = chatbot.get_relevant_documents(question)
                print(f"\n🔍 Retrieved {len(docs)} relevant documents:")
                for i, doc in enumerate(docs, 1):
                    print(f"\n{i}. {doc.meta.get('title', 'Untitled')}")
                    print(f"   Source: {doc.meta.get('source', 'Unknown')}")
                    print(f"   Content: {doc.content[:200]}...")
                continue
            elif not user_input:
                continue
            
            print("\n🤖 Bot: ", end="", flush=True)
            response = chatbot.chat(user_input)
            print(response)
    
    except Exception as e:
        print(f"\n❌ Error initializing chatbot: {e}")
        print("\nPlease ensure:")
        print("1. Ollama is installed and running (ollama serve)")
        print("2. Required model is available (ollama pull llama2)")
        print("3. All Python dependencies are installed")

if __name__ == "__main__":
    main()