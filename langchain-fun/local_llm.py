from langchain_community.llms import Ollama

try:
    llm = Ollama(model="gemma:2b", base_url='http://localhost:11434')
    prompt = ("Tell me about yourself!")
    response = llm.invoke(prompt)
    print(response)
except Exception as e:
    print(f"An unexpected error occurred: {e}")