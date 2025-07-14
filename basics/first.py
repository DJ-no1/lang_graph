from langchain_google_genai import GoogleGenerativeAI
import dotenv
from langgraph.graph import StateGraph,START, END
from typing import TypedDict


dotenv.load_dotenv()

class bmiDic(TypedDict):
    height: float
    weight: float
    bmi: float
    label: str

def cal_bmi(state : bmiDic) -> bmiDic:
    weight = state['weight']
    height = state['height']

    bmi = weight/(height**2)

    state['bmi'] = round(bmi, 2)

    return state

def label(state: bmiDic) -> bmiDic:
    if state['bmi'] < 18.5:
        state['label'] = 'Underweight'
    elif state['bmi'] < 24.9:
        state['label'] = 'Normal weight'
    elif state['bmi'] < 29.9:
        state['label'] = 'Overweight'
    else:
        state['label'] = 'Obesity'
    return state


# Graph defining 
graph = StateGraph(bmiDic)


# create nodes 
graph.add_node('bmi_calculator', cal_bmi)
graph.add_node('label', label)


# add edge
graph.add_edge(START, 'bmi_calculator' )
graph.add_edge('bmi_calculator', 'label')
graph.add_edge('label',END)



#compile 
workflow = graph.compile()

#Execute the graph
userinput= float(input("weight in kg: "))
userinput_for_height = float(input("height in meters: "))
initial_state = {'weight': userinput, 'height': userinput_for_height }

final_state = workflow.invoke(initial_state)

print(final_state)