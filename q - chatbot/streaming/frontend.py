import os
from dotenv import load_dotenv
import click
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables from .env
load_dotenv()

CHATBOT_NAME = os.getenv('CHATBOT_NAME', 'my-chatbot')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# In-memory chat history: {conversation_id: [ {'role': 'user'|'ai', 'message': str}, ... ] }
chat_histories = {}

# Gemini 2.5 Flash Lite via LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

def store_chat_history(conversation_id, role, message):
    if conversation_id not in chat_histories:
        chat_histories[conversation_id] = []
    chat_histories[conversation_id].append({'role': role, 'message': message})

def retrieve_chat_history(conversation_id):
    return chat_histories.get(conversation_id, [])

def format_history_for_llm(history):
    messages = []
    for entry in history:
        if entry['role'] == 'user':
            messages.append(HumanMessage(content=entry['message']))
        else:
            messages.append(AIMessage(content=entry['message']))
    return messages

def chat(conversation_id, user_message):
    store_chat_history(conversation_id, 'user', user_message)
    history = retrieve_chat_history(conversation_id)
    messages = format_history_for_llm(history)
    response = llm(messages)
    ai_message = response.content if hasattr(response, 'content') else str(response)
    store_chat_history(conversation_id, 'ai', ai_message)
    return ai_message

@click.group()
def cli():
    """Customizable Gemini 2.5 Flash Lite Chatbot CLI (in-memory, no DB)"""
    pass

@cli.command()
@click.option('--conversation-id', prompt=True, help='Conversation/thread ID')
@click.option('--message', prompt=True, help='Your message')
def send(conversation_id, message):
    """Send a message to the chatbot (with threading and history)."""
    response = chat(conversation_id, message)
    click.echo(f"AI: {response}")

@cli.command()
@click.option('--conversation-id', prompt=True, help='Conversation/thread ID')
def history(conversation_id):
    """Show chat history for a conversation/thread."""
    history = retrieve_chat_history(conversation_id)
    for entry in history:
        click.echo(f"{entry['role']}: {entry['message']}")

@cli.command()
def list_threads():
    """List all conversation/thread IDs."""
    for t in chat_histories.keys():
        click.echo(t)

@cli.command()
@click.option('--conversation-id', prompt=True, help='Conversation/thread ID')
def clear(conversation_id):
    """Clear chat history for a conversation/thread."""
    chat_histories.pop(conversation_id, None)
    click.echo(f"Cleared history for {conversation_id}")

if __name__ == '__main__':
    cli()
