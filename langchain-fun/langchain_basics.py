from langchain.chains.sequential import SequentialChain
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
    input_variables=['subject'],
    template="I want to write a poem about {subject}. Please suggest a good title."
)

name_chain = LLMChain(llm=client, prompt=prompt_template_name, output_key="poem_title")

prompt_template_items = PromptTemplate(
    input_variables=['poem_title'],
    template="Give me some beautiful lines for a poem with title {poem_title}."
)

poem_lines = LLMChain(llm=client, prompt=prompt_template_items, output_key="poem_lines")

chain = SequentialChain(
    chains=[name_chain, poem_lines],
    input_variables=['subject'],
    output_variables=['poem_title', "poem_lines"]
)

response = chain({'subject': 'philosophy'})
print(response)