import os
import sys
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

load_dotenv()
api_key = os.getenv('MISTRAL_API_KEY')
print(f'API Key initialized: {bool(api_key)}')

if not api_key:
    print("Error: No API key found.")
    sys.exit(1)

try:
    llm = ChatMistralAI(model='open-mistral-7b', api_key=api_key)
    print('Sending ping to Mistral AI via Langchain (non-streaming)...')
    response = llm.invoke([HumanMessage(content='Say Pong')])
    print(f'SUCCESS (invoke)! Response: {response.content}')
    
    print('\nTesting Streaming...')
    llm_stream = ChatMistralAI(model='open-mistral-7b', api_key=api_key, streaming=True)
    print('Sending ping to Mistral AI via Langchain (streaming)...')
    for chunk in llm_stream.stream([HumanMessage(content='Say Pong streaming')]):
        print(chunk.content, end='')
    print('\nSUCCESS (stream)!')
    
except Exception as e:
    print(f'\nFAILED: {type(e).__name__}: {str(e)}')
