import os
from langsmith import Client
from evalFunctions import TestEvaluator

'''
File to run experiments
'''

# initialise environmental values
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
print('Enter your LangChain API Key: ', end='')
key = input()
os.environ["LANGCHAIN_API_KEY"] = key

# initialise client
client = Client()


projects = ["<YOUR PROJECT NAMEs>"]

# These were the project names used for my experiments
# projects = ["Prompting Level 1 e82a6b7f-cfa9-4df1-97d5-0261273d77ae", 
#             "Prompting Level 2 c62bc917-2e17-460b-b8c9-45d7df9e3a32",
#             "Prompting Level 3 73db733d-ab32-4d5e-a4ea-9b13cf6d9f2d",
#             "Training on evalSlides 90b5f92f-8205-4223-9b1e-aaffd963c52e",
#             "Training on evalTranscripts 9d3f9831-095e-4b28-ad1b-4ee0893365d1",
#             "Models\Meta-Llama-3-8B-Instruct.Q5_K_M.gguf eefab801-3fae-494a-a8df-310a09d16a77",
#             "Models\zephyr-7b-beta.Q4_K_M.gguf e46ccb8b-7d6e-4ba7-a14d-7096dcc4e1d8",
#             "Models\Starling-LM-7B-beta-Q4_K_M.gguf 0840573d-4d84-42e2-b2fb-e12a9def3cd7"]

# obtain runs from Langsmith
runs = client.list_runs(
        project_name= projects,
        execution_order=1
    )

for run in runs:
    # initialise evaluator
    evaluator = TestEvaluator()

    # run evaluation
    for run in runs:
        print(run.id)
        examples = client.list_examples(example_ids=run.reference_example_id)
        for example in examples:
            ex = example
            print(ex)
        feedback = client.evaluate_run(run, evaluator=evaluator, reference_example=ex)
