from app.models import Agent
from app import db
import json
import logging
from langchain_groq import ChatGroq
from config import Config
import os
import shutil
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def extract_domain_keywords(description: str) -> List[str]:
    """Extract key domain terms from agent description"""
    system_message = """Extract key domain terms and concepts from the following agent description. 
    Return only the most relevant keywords that define the agent's specific domain of expertise.
    Format your response as a comma-separated list of keywords.
    Example: If the description is about a medical expert in cardiology, return: cardiology, heart disease, cardiovascular system, cardiac care
    """

    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": f"Extract domain keywords from this description: {description}"
        }
    ]

    try:
        llm = ChatGroq(
            model="llama-3.2-90b-vision-preview",
            temperature=0.3,
            groq_api_key=Config.GROQ_API_KEY
        )

        response = llm.invoke(messages)
        keywords = [kw.strip().lower() for kw in response.content.split(',')]
        logger.info(f"Extracted keywords: {keywords}")
        return keywords
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return []


def validate_agent_configuration(name: str, description: str, configurations: Dict) -> Tuple[bool, str]:
    """Validate agent configuration and ensure it meets requirements"""
    try:
        if not name or len(name.strip()) < 3:
            return False, "Agent name must be at least 3 characters long"

        if not description or len(description.strip()) < 10:
            return False, "Agent description must be at least 10 characters long"

        required_configs = ['model', 'temperature']
        for config in required_configs:
            if config not in configurations:
                return False, f"Missing required configuration: {config}"

        if not isinstance(configurations['temperature'], (int, float)) or \
                not 0 <= configurations['temperature'] <= 1:
            return False, "Temperature must be a number between 0 and 1"

        return True, "Configuration valid"
    except Exception as e:
        logger.error(f"Error validating agent configuration: {str(e)}")
        return False, str(e)


def create_agent_with_documents(name: str, description: str, configurations: Dict[str, Any], doc_dir: str,
                                user_id: int) -> Agent:
    """Create an agent with enhanced domain specificity"""
    try:
        # Validate configuration
        is_valid, message = validate_agent_configuration(name, description, configurations)
        if not is_valid:
            raise ValueError(message)

        # Extract domain keywords
        domain_keywords = extract_domain_keywords(description)
        if not domain_keywords:
            logger.warning("No domain keywords extracted, using fallback method")
            domain_keywords = [word.lower() for word in description.split() if len(word) > 3]

        # Enhance the agent's system prompt
        enhanced_description = generate_enhanced_description(description, domain_keywords)

        # Create enhanced configurations with base settings
        enhanced_configurations = {
            'model': configurations.get('model', "llama-3.2-90b-vision-preview"),
            'temperature': configurations.get('temperature', 0.7),
            'domain_keywords': domain_keywords,
            'strict_domain': True,
            'enhanced_description': enhanced_description,
            'collection_name': None  # Wird später durch load_documents gesetzt
        }

        # Create agent
        agent = Agent(
            name=name,
            description=description,
            configurations=json.dumps(enhanced_configurations),
            user_id=user_id,
            document_store=None  # Wird später durch load_documents gesetzt
        )

        db.session.add(agent)
        db.session.commit()

        # Create document directory for agent
        data_dir = f'data/agent_{agent.id}'
        os.makedirs(data_dir, exist_ok=True)

        # Copy documents to agent's directory
        if os.path.exists(doc_dir):
            for filename in os.listdir(doc_dir):
                src = os.path.join(doc_dir, filename)
                dst = os.path.join(data_dir, filename)
                shutil.copy2(src, dst)  # Use copy2 to preserve metadata
                logger.info(f"Copied {src} to {dst}")

        logger.info(f"Created agent: {name} with domain keywords: {domain_keywords}")
        return agent

    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        if 'data_dir' in locals() and os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        db.session.rollback()
        raise


def generate_enhanced_description(description: str, domain_keywords: List[str]) -> str:
    """Generate an enhanced description for the agent using LLM"""
    system_message = """Create an enhanced description for an AI agent based on the original description 
    and domain keywords. The enhanced description should clearly define the agent's:
    1. Specific domain of expertise
    2. Boundaries of knowledge
    3. Type of questions it can answer
    4. Approach to handling queries
    Format the response as a clear, professional description."""

    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": f"""Original Description: {description}
            Domain Keywords: {', '.join(domain_keywords)}

            Generate an enhanced description for this agent."""
        }
    ]

    try:
        llm = ChatGroq(
            model="llama-3.2-90b-vision-preview",
            temperature=0.4,
            groq_api_key=Config.GROQ_API_KEY
        )

        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error generating enhanced description: {str(e)}")
        return description  # Fallback to original description


def update_agent_configuration(agent: Agent, new_config: Dict[str, Any]) -> bool:
    """Update an agent's configuration while preserving essential settings"""
    try:
        current_config = json.loads(agent.configurations)

        # Preserve domain-specific settings
        domain_settings = {
            'domain_keywords': current_config.get('domain_keywords', []),
            'strict_domain': current_config.get('strict_domain', True),
            'enhanced_description': current_config.get('enhanced_description', ''),
            'collection_name': current_config.get('collection_name')
        }

        # Merge new configuration with preserved settings
        updated_config = {**new_config, **domain_settings}

        # Validate new configuration
        is_valid, message = validate_agent_configuration(
            agent.name,
            agent.description,
            updated_config
        )

        if not is_valid:
            logger.error(f"Invalid configuration update: {message}")
            return False

        # Update agent configuration
        agent.configurations = json.dumps(updated_config)
        db.session.commit()

        logger.info(f"Updated configuration for agent: {agent.name}")
        return True

    except Exception as e:
        logger.error(f"Error updating agent configuration: {str(e)}")
        db.session.rollback()
        return False


def process_agent_query(agent: Agent, query: str, conversation_history: List = None) -> str:
    """Process a query using the agent's configured model and settings"""
    try:
        config = json.loads(agent.configurations)
        model = config.get('model', "llama-3.2-90b-vision-preview")
        temperature = float(config.get('temperature', 0.7))

        logger.info(f"Processing query with model {model} and temperature {temperature}")

        # Use the process_query from rag.py with the agent's settings
        from app.rag import process_query
        return process_query(agent, query, conversation_history)

    except Exception as e:
        logger.error(f"Error in agent query processing: {str(e)}")
        return f"Error processing query: {str(e)}"
