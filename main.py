import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import re
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.responses import FileResponse

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, encoding='iso-8859-1')

    def clean_text(text):
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower().strip()
        return text

    df['cleaned_input'] = df['input'].apply(clean_text)
    df['cleaned_response'] = df['response'].apply(clean_text)

    df.to_csv("cleaned_conversations.csv", index=False)
    return df

def generate_response(prompt):
    model_name = "gpt2"
    model = TFGPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    inputs = tokenizer.encode(prompt, return_tensors="tf")
    attention_mask = tokenizer.encode_plus(prompt, return_tensors="tf")['attention_mask']
    eos_token_id = tokenizer.eos_token_id
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=1000,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        eos_token_id=eos_token_id,
        temperature=0.7,
        top_p=0.9
    )
    message = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # message = message.split("\n")[0]
    return message

app = FastAPI()

class ChatInput(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the chatbot API!"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse('favicon.ico')

@app.post("/chat")
def chat(input: ChatInput):
    prompt = f"User: {input.message}\nBot: "
    response = generate_response(prompt)
    return {"response": response}

def start_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

def main():
    load_and_clean_data('Conversation.csv')
    start_api()

if __name__ == "__main__":
    main()
