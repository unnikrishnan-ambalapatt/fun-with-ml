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