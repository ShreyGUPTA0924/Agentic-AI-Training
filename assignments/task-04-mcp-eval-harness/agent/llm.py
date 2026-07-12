import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def get_llm(temperature: float = 0.0, model: str = None):
    """Return a chat model based on the first available API key.
    
    Order of preference:
    1. Groq (using openai/gpt-oss-120b by default — chosen over Llama models on
       Groq because it produces reliably well-formed structured tool calls;
       Llama-3.3-70b-versatile was observed to occasionally emit malformed
       "<function=name>{args}</function>" text instead of a real tool call
       when many MCP tools are bound, which Groq's API then rejects)
    2. OpenAI (using gpt-4o-mini by default)
    3. Anthropic (using claude-3-5-sonnet-latest by default)
    4. Google GenAI (using gemini-1.5-flash by default)
    """
    if os.getenv("GROQ_API_KEY"):
        model_name = model or "openai/gpt-oss-120b"
        logger.info(f"Initializing ChatGroq ({model_name})")
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=model_name,
            temperature=temperature
        )
    elif os.getenv("OPENAI_API_KEY"):
        logger.info("Initializing ChatOpenAI (gpt-4o-mini)")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            temperature=temperature
        )
    elif os.getenv("ANTHROPIC_API_KEY"):
        logger.info("Initializing ChatAnthropic (claude-3-5-sonnet-latest)")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-5-sonnet-latest",
            temperature=temperature
        )
    elif os.getenv("GOOGLE_API_KEY"):
        logger.info("Initializing ChatGoogleGenAI (gemini-1.5-flash)")
        from langchain_google_genai import ChatGoogleGenAI
        return ChatGoogleGenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini-1.5-flash",
            temperature=temperature
        )
    else:
        raise ValueError("No LLM provider API key found (tried GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY).")
