# src/server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import asyncio
from agent import get_agent

app = FastAPI(title="Gemini LangGraph AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str
    thread_id: str = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    model: str = "gemini-2.0-flash"

agent = get_agent()

@app.get("/")
def root():
    return {"message": "Gemini LangGraph AI Agent API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "gemini-2.0-flash"}

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    try:
        # Invoke the Gemini agent
        result = agent.invoke({
            "messages": [("human", chat_message.message)]
        })
        
        # Extract the response
        response = result["messages"][-1].content
        
        return ChatResponse(
            response=response,
            thread_id=chat_message.thread_id or "default",
            model="gemini-2.0-flash"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.post("/chat/stream")
async def stream_chat(chat_message: ChatMessage):
    async def generate():
        try:
            # Stream the Gemini agent response
            for chunk in agent.stream(
                {"messages": [("human", chat_message.message)]}, 
                stream_mode="values"
            ):
                if "messages" in chunk:
                    latest_message = chunk["messages"][-1]
                    if hasattr(latest_message, 'content') and latest_message.content:
                        yield f"data: {json.dumps({
                            'content': latest_message.content, 
                            'type': 'message',
                            'model': 'gemini-2.0-flash'
                        })}\n\n"
            
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'type': 'error'})}\n\n"
    
    return StreamingResponse(
        generate(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
