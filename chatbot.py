from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import PersonalizationConfig, DEFAULT_CONFIG
from rag_engine import RAGEngine
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PersonalizedChatbot:
    def __init__(self, rag_engine: RAGEngine, config: PersonalizationConfig = DEFAULT_CONFIG):
        """
        Initialize the personalized chatbot.
        
        Args:
            rag_engine (RAGEngine): The RAG engine instance
            config (PersonalizationConfig): Personalization settings
        """
        self.rag_engine = rag_engine
        self.config = config
        
        # Initialize LLM with OpenRouter configuration
        try:
            self.llm = ChatOpenAI(
                model="mistralai/mistral-7b-instruct",  # Using Mistral model
                temperature=0.7,
                openai_api_base=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                default_headers={
                    "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:8501"),
                    "X-Title": os.getenv("SITE_NAME", "Personalized RAG Chatbot")
                },
                max_retries=3,
                timeout=60  # Increased timeout
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
        
        self.conversation_history: List[Dict] = []

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create a prompt template based on personalization settings."""
        system_template = f"""You are a helpful AI assistant with the following characteristics:
        - Tone: {self.config.tone}
        - Communication Goal: {self.config.communication_goal}
        - Response Style: {self.config.response_style}
        - Target Audience: {self.config.user_persona}
        - Language: {self.config.language}
        
        Use the following context to answer the user's question. If the context doesn't contain
        relevant information, say so and provide a general response based on your knowledge.
        
        Context: {{context}}
        
        Previous conversation:
        {{chat_history}}
        
        User question: {{question}}
        
        Provide a {self.config.response_length} response in a {self.config.response_style} style."""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{question}")
        ])

    def get_response(self, query: str) -> str:
        """
        Generate a personalized response based on the query and context.
        
        Args:
            query (str): User's question
            
        Returns:
            str: Generated response
        """
        try:
            # Retrieve relevant context
            try:
                context = self.rag_engine.retrieve_relevant_chunks(query)
                context_text = "\n\n".join(context)
                logger.info(f"Retrieved context: {context_text[:100]}...")  # Log first 100 chars
            except ValueError as e:
                logger.warning(f"No relevant documents found: {str(e)}")
                context_text = "No relevant documents have been processed yet."

            # Format chat history
            chat_history = "\n".join([
                f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                for msg in self.conversation_history[-5:]  # Keep last 5 exchanges
            ])

            # Create and format prompt
            prompt = self._create_prompt_template()
            formatted_prompt = prompt.format_messages(
                context=context_text,
                chat_history=chat_history,
                question=query
            )
            logger.info(f"Formatted prompt: {formatted_prompt}")

            # Generate response with error handling
            try:
                logger.info("Attempting to generate response...")
                response = self.llm.invoke(formatted_prompt)
                logger.info(f"Raw response: {response}")
                
                if not response:
                    logger.error("Empty response received")
                    return "I apologize, but I received an empty response. Please try again."
                
                if not hasattr(response, 'content'):
                    logger.error(f"Response missing content attribute: {response}")
                    return "I apologize, but I received an invalid response format. Please try again."
                
                if not response.content:
                    logger.error("Response content is empty")
                    return "I apologize, but I received an empty response. Please try again."
                
                # Update conversation history
                self.conversation_history.append({
                    "user": query,
                    "assistant": response.content
                })
                logger.info("Response generated and conversation history updated successfully")

                return response.content
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}", exc_info=True)
                error_msg = f"I apologize, but I encountered an error while generating a response: {str(e)}"
                self.conversation_history.append({
                    "user": query,
                    "assistant": error_msg
                })
                return error_msg
                
        except Exception as e:
            logger.error(f"Unexpected error in get_response: {str(e)}", exc_info=True)
            return f"I apologize, but I encountered an unexpected error: {str(e)}"

    def update_config(self, new_config: PersonalizationConfig) -> None:
        """
        Update the personalization configuration.
        
        Args:
            new_config (PersonalizationConfig): New configuration settings
        """
        self.config = new_config

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = [] 