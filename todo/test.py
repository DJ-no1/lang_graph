from langchain_google_genai import GoogleGenerativeAI
import dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
import datetime
dotenv.load_dotenv()

# model initialization
model = GoogleGenerativeAI(
    model = "gemini-2.5-flash"
    )

class todoStructure(TypedDict):
    taskid: str
    title: str
    description: str
    due_date: datetime.datetime | None
    priority: bool | None
    category: Literal["work", "college", "personal"] | None
    status: Literal["havent started", "in progress", "completed"] | None

class todoSummary(TypedDict):
    taskid: str
    task_description: str

class todoCommandDict(TypedDict):
    taskid: str
    title: str
    description: str
    due_date: datetime.datetime | None
    priority: bool | None
    category: Literal["work", "college", "personal"] | None
    status: Literal["havent started", "in progress", "completed"] | None
    command: Literal["add", "update", "remove"]

class todoState(TypedDict):
    userInput: str
    taskList: list[todoStructure]
    summaryDict: dict[str, str]  # taskid -> description
    commandList: list[todoCommandDict]

def extract_tasks_node(state: todoState) -> todoState:
    import json
    import re
    userInput = state["userInput"]
    prompt = (
        "Extract tasks from the following conversation. For each task, generate a dictionary with: "
        "taskid (unique), title, concise description, due_date (if mentioned), priority (if mentioned), "
        "category (work/college/personal), status (havent started/in progress/completed). "
        "Output ONLY a valid JSON array of such dictionaries, nothing else.\nConversation: "
        f"{userInput}"
    )
    raw_output = model.invoke(prompt)
    print('DEBUG raw_output:', raw_output)  # Debug print
    if isinstance(raw_output, str):
        # Remove triple backticks and 'json' label
        cleaned = re.sub(r"^```json|^```|```$", "", raw_output, flags=re.MULTILINE).strip()
        # Extract the first JSON array in the string
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            json_str = cleaned
        try:
            taskList = json.loads(json_str)
            if not isinstance(taskList, list):
                print('DEBUG: Parsed JSON is not a list, got:', type(taskList))
                taskList = []
        except Exception as e:
            print('DEBUG: JSON parsing error:', e)
            taskList = []
    else:
        taskList = raw_output if isinstance(raw_output, list) else []
    state["taskList"] = taskList
    return state

def summarize_node(state: todoState) -> todoState:
    # Create summaryDict: taskid -> description or concise_description
    summaryDict = {task["taskid"]: task.get("description", task.get("concise_description", "")) for task in state["taskList"]}
    state["summaryDict"] = summaryDict
    return state

def compare_node(state: todoState) -> todoState:
    import os, json
    save_path = os.path.join(os.path.dirname(__file__), "save.txt")
    previous_summary = {}
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            try:
                previous_summary = json.load(f)
            except Exception:
                previous_summary = {}
    commandList = []
    # Add/update tasks
    for task in state["taskList"]:
        desc = task.get("description", task.get("concise_description", ""))
        prev_desc = previous_summary.get(task["taskid"])
        if prev_desc is None:
            cmd = "add"
        elif prev_desc != desc:
            cmd = "update"
        else:
            cmd = "none"
        if cmd != "none":
            # Ensure 'description' key is present for downstream nodes
            task_with_desc = dict(task)
            task_with_desc["description"] = desc
            commandList.append({**task_with_desc, "command": cmd})
    # Remove tasks not in new list
    for prev_id in previous_summary:
        if prev_id not in state["summaryDict"]:
            commandList.append({"taskid": prev_id, "title": "", "description": previous_summary[prev_id], "due_date": None, "priority": None, "category": None, "status": None, "command": "remove"})
    # Update summaryDict for next run
    state["summaryDict"].update({task["taskid"]: task.get("description", task.get("concise_description", "")) for task in state["taskList"]})
    state["commandList"] = commandList
    return state

def execute_node(state: todoState) -> todoState:
    import os, json
    save_path = os.path.join(os.path.dirname(__file__), "save.txt")
    # Load previous file if exists
    previous_data = {"todos": [], "summary": {}}
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            try:
                previous_data = json.load(f)
            except Exception:
                previous_data = {"todos": [], "summary": {}}
    # Update summary dictionary
    summary_dict = previous_data.get("summary", {})
    for cmd_dict in state["commandList"]:
        cmd = cmd_dict["command"]
        tid = cmd_dict["taskid"]
        if cmd == "add" or cmd == "update":
            summary_dict[tid] = cmd_dict["description"]
        elif cmd == "remove":
            summary_dict.pop(tid, None)
    # Update todos list
    todos = previous_data.get("todos", [])
    # Remove todos marked for removal
    todos = [todo for todo in todos if todo.get("taskid") not in [cmd["taskid"] for cmd in state["commandList"] if cmd["command"] == "remove"]]
    # Add/update todos
    for cmd_dict in state["commandList"]:
        if cmd_dict["command"] in ["add", "update"]:
            # Remove old todo with same id if exists
            todos = [todo for todo in todos if todo.get("taskid") != cmd_dict["taskid"]]
            todos.append({k: v for k, v in cmd_dict.items() if k != "command"})
    # Save both todos and summary
    with open(save_path, "w") as f:
        json.dump({"todos": todos, "summary": summary_dict}, f, indent=2)
    return state

# defining graph
graph = StateGraph(todoState)

# adding nodes
graph.add_node('extractTasks', extract_tasks_node)
graph.add_node('summarize', summarize_node)
graph.add_node('compare', compare_node)
graph.add_node('execute', execute_node)

# adding edges
graph.add_edge(START, 'extractTasks')
graph.add_edge('extractTasks', 'summarize')
graph.add_edge('summarize', 'compare')
graph.add_edge('compare', 'execute')
graph.add_edge('execute', END)

# compile graph
workflow = graph.compile()

# Execute the graph
initial_state = {
    'userInput': "I bought groceries, called dad, and finished my project report on biology . i need to gift my sister a cake"
}
final_state = workflow.invoke(initial_state)
print(final_state)