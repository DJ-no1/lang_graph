from langchain_google_genai import GoogleGenerativeAI
import dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
dotenv.load_dotenv()


model = GoogleGenerativeAI(
    model = "gemini-2.5-flash"
    )

class outlineState(TypedDict):
    title: str
    outline: str
    blog : str
    evaluation: str


def outline(state: outlineState) -> outlineState:
    title = state["title"]
    prompt = f"Create an detailed outline for a document titled '{title}'."
    outline = model.invoke(prompt)

    #update state with the response

    state["outline"] = outline

    #saving the outline 


    return state


def blog(state: outlineState) -> outlineState:
    outline = state["outline"]
    prompt = f"Create a blog post based on the following outline: {outline}"
    blog = model.invoke(prompt)

    state["blog"] = blog

    return state

def evaluation(state: outlineState) -> outlineState:
    outline = state["outline"]
    blog = state["blog"]
    prompt = f"Evaluate the following blog post based on the outline: {outline}\n\nBlog Post:\n{blog} and provide a score from 1 to 10."
    evaluation = model.invoke(prompt)

    state["evaluation"] = evaluation

    return state

userinput = input("Enter the title of your document: ")


#define graph 
graph = StateGraph(outlineState)


# create nodes
graph.add_node('outline', outline)
graph.add_node('blog', blog)
graph.add_node('evaluation', evaluation)

# add edge 
graph.add_edge(START, 'outline')
graph.add_edge('outline', 'blog')
graph.add_edge('blog', 'evaluation')
graph.add_edge('evaluation', END)


#compile the graph
workflow = graph.compile()

#execute the graph

initial_state = {'title': userinput}

final_state = workflow.invoke(initial_state)

print(final_state["outline"])
print(final_state["blog"])
print(final_state["evaluation"])