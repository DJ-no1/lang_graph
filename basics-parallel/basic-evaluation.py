from langchain_google_genai import GoogleGenerativeAI
import dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

model = GoogleGenerativeAI(
    model="gemini-2.5-flash"
)

class questionState(TypedDict):
    userInput: str
    question: str
    detailedQuestion: str
    lessDetailedQuestion: str


def question_node(state: questionState) -> questionState:
    userInput = state["userInput"]
    prompt = f"identify the main question in the following conversation: {userInput}. "
    normal_question = model.invoke(prompt)
    
    return {'question': normal_question}

def detailed_question_node(state: questionState) -> questionState:
    question = state["question"]
    prompt = f"Rewrite the Question in more abstract way: {question}. "
    detailed_question = model.invoke(prompt)
    return {'detailedQuestion': detailed_question}

def less_detailed_question_node(state: questionState) -> questionState:
    question = state["question"]
    prompt = f"Rewrite the Question in less abstract way: {question}. "
    less_detailed_question = model.invoke(prompt)
    return {'lessDetailedQuestion': less_detailed_question}

# defining graph
graph = StateGraph(questionState)

# adding nodes
graph.add_node('question', question_node)
graph.add_node('detailed_question', detailed_question_node)
graph.add_node('less_detailed_question', less_detailed_question_node)

# adding edges
graph.add_edge(START, 'question')
graph.add_edge('question', 'detailed_question')
graph.add_edge('question', 'less_detailed_question')
graph.add_edge('question', END)
graph.add_edge('detailed_question', END)
graph.add_edge('less_detailed_question', END)

# compile the graph
workflow = graph.compile()

# executing the graph

initial_state = {'userInput': input("Enter your conversation: ")}
final_state = workflow.invoke(initial_state)

print(final_state)