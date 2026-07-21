import os
import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# AI Pipe OpenAI-compatible client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

app = FastAPI()

# CORS for IITM grader
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DynamicRequest(BaseModel):
    text: str
    schema: dict

def build_prompt(text: str, schema: dict):
    return f"""
You are a strict JSON extractor.

Extract ONLY the fields listed in the schema.
Use EXACTLY the same keys.
If a value is missing, return null.
Never invent extra keys.
Return ONLY valid JSON.

TEXT:
{text}

SCHEMA:
{json.dumps(schema, indent=2)}

Return JSON ONLY.
"""

def convert_value(value, typ):
    if value is None:
        return None

    try:
        if typ == "string":
            return str(value)

        if typ == "integer":
            return int(value)

        if typ == "float":
            return float(value)

        if typ == "boolean":
            if isinstance(value, bool):
                return value
            if str(value).lower() in ["true", "yes", "1"]:
                return True
            if str(value).lower() in ["false", "no", "0"]:
                return False
            return None

        if typ == "date":
            # Expect YYYY-MM-DD
            try:
                dt = datetime.strptime(value, "%Y-%m-%d")
                return dt.strftime("%Y-%m-%d")
            except:
                return None

    except:
        return None

    return None

@app.post("/dynamic-extract")
def dynamic_extract(payload: DynamicRequest):
    prompt = build_prompt(payload.text, payload.schema)

    # Call AI Pipe
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # AI Pipe supports this
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content

    # Parse JSON safely
    try:
        extracted = json.loads(raw)
    except:
        extracted = {}

    # Build strict output
    final = {}

    for key, typ in payload.schema.items():
        value = extracted.get(key, None)
        final[key] = convert_value(value, typ)

    return final
