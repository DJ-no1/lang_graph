from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
import datetime
dotenv.load_dotenv()

savedList = {'savedList': 'Here is your task list:\n\n*   Buy groceries\n*   Call mom\n*   Finish project report on operating system\n*   Finish laundry\n*   Call manager about meeting', 'category': "Here's the categorization of your tasks:\n\n**Work:**\n*   Call manager about meeting\n\n**College:**\n*   Finish project report on operating system\n\n**Personal:**\n*   Buy groceries\n*   Call mom\n*   Finish laundry"}

# model initialization
model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
    )

## add structured output for model
## use conditional flow 


class todoStructure(TypedDict):
    taskid: str
    title: str
    description: str 
    due_date:  datetime.datetime | None
    priority: bool | None
    category: Literal["work", "college", "personal"] | None
    status: Literal["havent started", "in progress", "completed"] | None


class todoSummary(TypedDict):
    taskid: todoStructure.taskid
    task_description: todoStructure.description

class todoState(TypedDict):
    userInput: str
    taskList: list[str]
    category: Literal["work", "college", "personal"] | None
    savedList: dict[str, str]

def task_List(state: todoState) -> todoState:
    userInput = state["userInput"]
    prompt = f"Create a task list based on the following conversation: {userInput} , make sure the tasks from the conversation only , do not add any extra tasks. "
    taskList = model.invoke(prompt)
    state["taskList"] = taskList
    return state

def categoryList(state: todoState) -> todoState:
    taskList = state["taskList"]
    prompt = f"Categorize the following tasks into work, college, and personal categories: {taskList} , make sure to categorize each task only once. "
    category = model.invoke(prompt)
    state["category"] = category
    return state


# defining graph
graph = StateGraph(todoState)

#adding nodes
graph.add_node('taskList', task_List)
graph.add_node('categoryList', categoryList)

#adding edges
graph.add_edge(START, 'taskList')
graph.add_edge('taskList', 'categoryList')
graph.add_edge('categoryList', END)

# compile graph
workflow = graph.compile()


# Execute the graph

initial_state = {'userInput': "“I bought my groceries, called, and finish my project report on operating system by tomorrow. I also need to finish my laundry later today. and call my manager about the meeting tomorrow."}

final_state = workflow.invoke(initial_state)

print(final_state)