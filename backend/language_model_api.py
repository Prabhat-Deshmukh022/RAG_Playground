from typing import Any, Optional
import requests
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

def language_model_api(option:int,prompt:str) -> Any:
    api_key=None
    match option:
        case 1:
            api_key=GEMINI_API_KEY
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent",
                # "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                # "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro-latest:generateContent",
                headers={
                    "Content-Type": "application/json"
                },
                params={
                    "key": api_key  # Make sure this is a valid Gemini API key
                },
                json={
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt}
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": 3000,
                        # "stopSequences": ["\n\n"]
                    }
                },
                timeout=10
            )

            response.raise_for_status()  # Raises exception for 4XX/5XX responses
            result = response.json()
            answer=result.get("candidates")[0].get("content")['parts'][0]['text']

            return answer
        
        case 2:
            api_key=MISTRAL_API_KEY

            headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
            }

            payload = {
                "model": "mistral-large-2411",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0
            }

            response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()

            answer = response.json()["choices"][0]["message"]["content"]

            return answer
                
        case 3:
            api_key=GROQ_API_KEY

            header = {
                "Authorization":f"Bearer {api_key}",
                "Content-Type":"application/json"
            }

            payload={
                "model":"llama-3.3-70b-versatile",
                "messages":[{
                    "role":"user",
                    "content":prompt
                }]
            }

            response = requests.post("https://api.groq.com/openai/v1/chat/completions",headers=header,json=payload)
            response.raise_for_status()

            answer = response.json()["choices"][0]["message"]["content"]

            return answer


