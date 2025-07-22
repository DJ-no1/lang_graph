
user_input = input("You: ")
state['user_input'] = user_input
state = workflow.invoke(state)
print("AI:", state['response'])
