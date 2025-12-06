from openai import OpenAI
from dotenv import load_dotenv
import os

with open('resources_data.md', 'r') as f:
    resources_text = f.read()

load_dotenv()

client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=os.getenv('api_key'),
)

def get_chat_response(text, health, subject):
    prompt = f"""
    You are a supportive student assistant.
    User message: "{text}"
    Detected mental state: {health}
    Detected subject: {subject}
    
    Use the available resources from the file below for tips: {resources_text}

    Generate:
    - empathetic emotional support (1 short paragraph)
    - practical, specific study advice (1 paragraph)
    - additional resources (bullet list)
    - where to find tutors if needed
    - Metropolia internal resources
    
    Format the response in Markdown with clear headings, separate paragraphs, and bulleted lists for resources.
    """

    response =  client.chat.completions.create(
        messages=[
            {"role": "user",
             "content": prompt}
        ],
        model="openai/gpt-4o",
        top_p=1,         # randomness
        max_tokens=500,  # max length
        temperature=0.7  # creativity
    )

    generated_text = response.choices[0].message.content.strip()

    # print(generated_text)
    return generated_text

if __name__ == '__main__':
    print(get_chat_response("help me with math task", "normal", "math"))