# How well can Large Language Models Learn?

## About
This project aims to assess the capabilities of Large Language Models (LLMs) in a learning environment such as classrooms. The LLMs will take in lecture slides from a particular module and attempt to answer exam like questions based on the topics in the lecture slides provided. 

## Setting up Requirements 
1. Install the required dependencies using this command. 
```sh
pip install -r requirements.txt
```

## Source File Breakdown
1. ChatBot interface
    - Run the `agents.py` file to launch the interface
    - To use the interface, upload a pdf file containing highlitable text to the documents section and type in a question regarding the document uploaded before hitting submit
    - Wait for the program to process your file before providing an answer

2. Experiments
    - The experimental data was created using `modelData.py`, `promptData.py`, and `trainingData.py`.
    - After creating the data, run `experiments.py` to run the evaluations on the projects you've created 
    - The evaluation functions and LLM is stored in `evalFunctions.py`

3. Context
    - `evalSlides`, `evalTranscripts` and `lectures` contain the chroma which the model uses to perform retrieval augmented generation. 

4. Downloading the LLMs
    - To download the LLMs, run `modelDownload.py` to download the model .gguf files in this directory. 
