from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
import dotenv
from pydantic import BaseModel , Field

from typing import TypedDict, Literal, Annotated
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
dotenv.load_dotenv()

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Define the state for the chat
class ChatState(TypedDict):
    user_input: str
    messages: list[HumanMessage | SystemMessage | AIMessage]
    response: str
    sentiment: Literal["continue", "stop"] = "continue"

class SentimentAnalysisResult(BaseModel):
    sentiment: Literal["continue", "stop"] = Field(..., description="Sentiment of the user input, if he/she wants to continue or stop the chat")


sentimentModel = model.with_structured_output(SentimentAnalysisResult)

def chatbot(state: ChatState) -> ChatState:
    # Add the user message to the state
    user_message = HumanMessage(content=state['user_input'])
    state['messages'].append(user_message)
    # Generate a response using the model (using message history)
    response = model.invoke(state['messages']).content
    state['messages'].append(AIMessage(content=response))
    state['response'] = response
    return state

def analyze_sentiment(state: ChatState) -> ChatState:
    # Analyze the sentiment of the user input
    sentiment_result = sentimentModel.invoke(f"check the sentiment of the user input if the user wants to continue or stop the chat, if the user wants to stop the chat, return 'stop', otherwise return 'continue'. User input: {state['user_input']}")
    # Update the state with the actual string value
    state['sentiment'] = sentiment_result.sentiment
    return state

def conditional_response(state: ChatState) :
    if state['sentiment'] == 'continue':
        return 'chatbot'
    else:
        return 'end'


# Define the graph
chat_graph = StateGraph(ChatState)

chat_graph.add_node('chatbot', chatbot)
chat_graph.add_node('analyze_sentiment', analyze_sentiment)

# Add end node
def end_conversation(state: ChatState) -> ChatState:
    print("Goodbye!")
    return state
chat_graph.add_node('end', end_conversation)

# adding edge
chat_graph.add_edge(START, 'analyze_sentiment')
chat_graph.add_conditional_edges('analyze_sentiment', conditional_response)
chat_graph.add_edge('end', END)

workflow = chat_graph.compile()

# Example usage: multi-turn chat
if __name__ == "__main__":
    # Initialize state with system message and empty messages list
    state: ChatState = {
        'user_input': '',
        'messages': [SystemMessage(content="You are a helpful AI assistant. Answer the user's questions clearly and concisely.")],
        'response': '',
        'sentiment': 'continue'
    }

    while state['sentiment'] != 'stop':
        user_input = input("You: ")
        state['user_input'] = user_input
        state = workflow.invoke(state)
        if state['sentiment'] != 'stop':
            print("AI:", state['response'])
    # The workflow will end via the 'end' node, which prints the goodbye message
