from langsmith.evaluation import EvaluationResult, RunEvaluator
from langsmith.evaluation.evaluator import EvaluationResults
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langsmith.schemas import Example, Run
from typing import Optional, Union
import re

'''
This file contains the evaluation functions
'''

def get_grade(criteria, result):
    '''
    Function to obtain standalone score from the evaluation results
    
    Args:
        criteria: keywords to find in the evaluation result
        result: the evaluation result
        
    Returns:
        mark: final marks/score found following the criteria
    '''
    for line in result.split("\n"):
        if f"{criteria}" in line:
            gradeline = line.split(f"{criteria}")[-1]
            match = re.findall("\d\.?\d?", gradeline)[0]
            if match:
                print(f"{criteria} is {match}")
                return float(match)
            else:
                print(f"{criteria} is N/A")


class TestEvaluator(RunEvaluator):

    def __init__(self):
        llm = LlamaCpp(model_path="Models\Starling-LM-7B-beta-Q4_K_M.gguf", 
                       n_ctx=50000, 
                       n_batch=512, 
                       max_tokens=-1,
                       verbose=True, 
                       stop=["QUESTION: ", "##","<|user|>"])

        template ="""GPT4 Correct User: 
        You are a lecturer marking exam questions from students.
        You will provide marks based on how closely the student is able to answer in comparison to the marking scheme provided.  
        The total score a student can achieve is described in the question. 
        You cannot award marks that are more than the total marks allocated for the question.
        Now, here is the question and marking scheme to it.

        Question: "{query}"

        Ideal Answer: "{answer}"

        Based on the information above, you are grading the student's answer, based on the question and marking scheme:

        The Answer To Grade: "{result}"

        Use the marking scheme to award marks to the student's answer.
        You MUST deduct 1 mark if the student answers in a vague manner or goes off topic.
        You MUST deduct 1 mark if the student answers over-explains the answer which means they spent too much time on the question. 
        Provide an explanation at the end for the marks awarded.
        The marks MUST be written in the format:
        
        Marks: (Final marks obtained from the answer) 
        Explanation: (Explanation for marks given)

        <|end_of_turn|>
        GPT4 Correct Assistant: Results:
        """

        self.eval_chain = PromptTemplate(input_variables=["query", "answer", "result"], template=template) | llm

    def evaluate_run(self, run: Run, example: Optional[Example] = None) -> Union[EvaluationResult, EvaluationResults]:
        '''
        Obtains the total mark obtained by the model.
        
        Args:
            run: Langsmith run
            example: corresponding example for Langsmith run
        ---
        Returns:
            mark: Total marks achieved from the answer given by the model
        '''

        if run.outputs is None:
            raise ValueError("Run outputs cannot be None")
        prediction = str(next(iter(run.outputs.values())))
        print(f"Evaluating run:  {run.id}")
        print(f"Input: {input}")
        # print(f"example: {example}")
        print(f"Predicted Answer: {prediction}")

        evaluator_result = self.eval_chain.invoke(
            {"query": input, "result": prediction, "answer":example}
        )
        print("result")
        print(evaluator_result)

        score = get_grade("Marks:", evaluator_result)
        
        if score is None:
            score = get_grade("Marks Obtained:", evaluator_result)
        
        if score is None:
            score = get_grade("Grade:", evaluator_result)
            
        if score is None:
            score = get_grade("marks obtained:", evaluator_result)
            
        print(f"Score is {score}")
        return EvaluationResults(results=[EvaluationResult(key="Score", score=score, comment=evaluator_result)])
