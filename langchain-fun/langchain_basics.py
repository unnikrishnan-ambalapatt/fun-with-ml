from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
    model="davinci-002",
    messages=[
        {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
    ]
)
print(completion.choices)
print("============================")
print(completion.choices[0].message)

prompt_template_name = PromptTemplate(
    input_variables=['cuisine'],
    template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
)

name_chain = LLMChain(llm=client, prompt=prompt_template_name, output_key="restaurant_name")
