import uvicorn
from fastapi import FastAPI, HTTPException
from transformers import pipeline, T5Tokenizer
from fastapi.middleware.cors import CORSMiddleware
import gc

app = FastAPI()

# Define allowed origins
origins = [
    "http://localhost:3000",  # Frontend URL
]

# Add the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the t5-base model
summarizer = pipeline("summarization", model="t5-base")

def split_text(text, max_tokens=512):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i+max_tokens]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    
    return chunks

@app.post("/summarize")
async def summarize(text: str, max_length: int = 100, min_length: int = 30):
    try:
        chunks = split_text(text)
        summary = ""
        for chunk in chunks:
            summary += summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()  # Ensure memory is freed after processing

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
