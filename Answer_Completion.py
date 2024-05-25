import openai

# enter your key
openai.api_key = 'API_KEY'

prompt = "You’re an assignment solving…."

response = openai.Completion.create(
  engine="text-davinci-003",
  prompt=prompt,
  temperature=0.7,
  max_tokens=50
)

print(response.choices[0].text.strip())
