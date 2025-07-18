from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict , Literal
import dotenv
from pydantic import BaseModel , Field
dotenv.load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# for model initialization this is just for creating a structured model
class sentimentState(BaseModel):
    sentiment: Literal["positive", "negative", ] = Field(description="The sentiment of the review")


# for model to understand the diagnosis we will use a structured model
class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')

    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')

    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')  


# we will use this state to track the review process

class ReviewState(TypedDict):

    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str


# creating a structured model for sentiment analysis
sentimentModal = model.with_structured_output(sentimentState)

# creating a structured model for diagnosis
diagnosisModal = model.with_structured_output(DiagnosisSchema)


def find_sentiment(state: ReviewState) -> ReviewState:

    prompt = f'For the following review find out the sentiment \n {state["review"]}'
    sentiment = sentimentModal.invoke(prompt).sentiment

    return {'sentiment': sentiment}  

def run_diagnosis(state: ReviewState) -> ReviewState:

    prompt = f"""Diagnose this negative review:\n\n{state['review']}\n"
    "Return issue_type, tone, and urgency.
"""
    response = diagnosisModal.invoke(prompt)

    return {'diagnosis': response.model_dump()}



def positive_response(state: ReviewState)-> ReviewState:

    prompt = f"""Write a short warm thank-you message in response to this review:
    \n\n\"{state['review']}\"\n
Also, kindly ask the user to leave feedback on our website."""
    
    response = model.invoke(prompt).content

    return {'response': response}


def negative_response(state: ReviewState):

    diagnosis = state['diagnosis']

    prompt = f"""You are a support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
Write an empathetic, helpful resolution message.
"""
    response = model.invoke(prompt).content

    return {'response': response}

def conditional_response(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:
    if state['sentiment'] == 'positive':
        return 'positive_response'
    else:
        return 'run_diagnosis'

# defining the graph 

graph = StateGraph(ReviewState)

# adding nodes

graph.add_node('find_sentiment', find_sentiment)
graph.add_node('run_diagnosis', run_diagnosis)
graph.add_node('negetive_response', negative_response)
graph.add_node('positive_response', positive_response)



# adding edges 

graph.add_edge(START, 'find_sentiment')
graph.add_conditional_edges('find_sentiment', conditional_response)
graph.add_edge('positive_response', END)
graph.add_edge('run_diagnosis', 'negetive_response')
graph.add_edge('negetive_response', END)


# compile graph 
workflow = graph.compile()

# execute the graph with initial state 

initial_state = {"review": "I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."}

final_state = workflow.invoke(initial_state)

print(final_state)  # Should print the final state with sentiment