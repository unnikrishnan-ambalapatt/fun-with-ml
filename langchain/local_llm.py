from langchain_community.llms import Ollama
import logging

try:
    llm = Ollama(model="gemma:2b")
    prompt = ("Tell me about yourself!")
    response = llm.invoke(prompt)
    print(response)
except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")