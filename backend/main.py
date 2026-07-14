import os
import tempfile
import uuid
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import json
import httpx
import asyncio
from dotenv import load_dotenv
import librosa
from scorer import init_models, evaluate_pipeline
load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
rooms: Dict[str, List[WebSocket]] = {}
client_user_map: Dict[WebSocket, str] = {}
room_reports: Dict[str, Dict[str, dict]] = {}

@app.on_event('startup')
async def startup_event():
    print('Initializing heavy ML models...')
    init_models()
    print('ML Models initialization complete.')

@app.post('/create-room')
async def create_room():
    """Generates and returns a unique room ID."""
    return {"roomId": str(uuid.uuid4())}

@app.websocket('/ws/{room_id}')
async def websocket_endpoint(websocket: WebSocket, room_id: str, userId: str=None):
    await websocket.accept()
    if room_id not in rooms:
        rooms[room_id] = []
        room_reports[room_id] = {}
    if len(rooms[room_id]) >= 2:
        await websocket.send_text(json.dumps({'type': 'error', 'message': 'Room is full'}))
        await websocket.close()
        return
    rooms[room_id].append(websocket)
    client_user_map[websocket] = userId
    for client in rooms[room_id]:
        if client != websocket:
            await client.send_text(json.dumps({'type': 'user-joined'}))
    try:
        while True:
            data = await websocket.receive_text()
            for client in rooms[room_id]:
                if client != websocket:
                    await client.send_text(data)
    except WebSocketDisconnect:
        if websocket in rooms.get(room_id, []):
            rooms[room_id].remove(websocket)
        if websocket in client_user_map:
            del client_user_map[websocket]
        if not rooms.get(room_id):
            if room_id in rooms:
                del rooms[room_id]
            if room_id in room_reports:
                del room_reports[room_id]
        else:
            for client in rooms[room_id]:
                await client.send_text(json.dumps({'type': 'user-left'}))

async def send_webhook(room_id: str, user_id: str, report: dict):
    webhook_url = os.environ.get("MAIN_BACKEND_WEBHOOK_URL")
    if not webhook_url:
        print("Webhook URL not configured, skipping webhook.")
        return
    
    payload = {
        "roomId": room_id,
        "userId": user_id,
        "report": report
    }
    
    async with httpx.AsyncClient() as client:
        try:
            print(f"Sending webhook to {webhook_url} for user {user_id} in room {room_id}")
            response = await client.post(webhook_url, json=payload, timeout=10.0)
            response.raise_for_status()
            print("Webhook sent successfully.")
        except Exception as e:
            print(f"Failed to send webhook: {e}")

def process_audio_sync(temp_file_path: str, question: str):
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is missing or invalid in .env")

        with open(temp_file_path, "rb") as audio_file:
            response = httpx.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                data=[
                    ("model", "whisper-large-v3"),
                    ("response_format", "verbose_json"),
                    ("timestamp_granularities[]", "word"),
                    ("timestamp_granularities[]", "segment")
                ],
                files={"file": (os.path.basename(temp_file_path), audio_file, "audio/webm")},
                timeout=60.0
            )
            response.raise_for_status()
            result = response.json()

        transcript_text = result.get("text", "").strip()
        segments_list = result.get("segments", [])
        word_timestamps_list = result.get("words", [])

        audio_array, sr = librosa.load(temp_file_path, sr=16000)
        
        report = evaluate_pipeline(transcript=transcript_text, segments=segments_list, word_timestamps=word_timestamps_list, audio_path=temp_file_path, audio_array=audio_array, sample_rate=sr, lang='en-US', question=question)
        return report
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'fluency': 0.0, 'lexical': 0.0, 'grammar': 0.0, 'pronunciation': 0.0, 'overall': 0.0, 'user_input': transcript_text if 'transcript_text' in locals() else 'N/A', 'feedback': {'fluency': f'Pipeline Error: {str(e)}', 'lexical': f'Pipeline Error: {str(e)}', 'grammar': f'Pipeline Error: {str(e)}', 'pronunciation': f'Pipeline Error: {str(e)}'}}

@app.post('/evaluate')
async def evaluate_audio(audio: UploadFile=File(...), roomId: str=Form(...), userId: str=Form(...), question: str=Form(None), background_tasks: BackgroundTasks=BackgroundTasks()):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f'{uuid.uuid4()}_{audio.filename}')
    with open(temp_file_path, 'wb') as buffer:
        buffer.write(await audio.read())
    file_size = os.path.getsize(temp_file_path)
    print(f'Received audio from {userId}, size: {file_size} bytes, content_type: {audio.content_type}')

    def cleanup():
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    background_tasks.add_task(cleanup)
    # Run the heavy AI models in a separate thread so we don't block other users
    report = await asyncio.to_thread(process_audio_sync, temp_file_path, question)
    if roomId not in room_reports:
        room_reports[roomId] = {}
    room_reports[roomId][userId] = report
    
    # Trigger the webhook asynchronously
    background_tasks.add_task(send_webhook, roomId, userId, report)
    
    if roomId in rooms:
        broadcast_data = json.dumps({'type': 'evaluation-ready', 'reports': room_reports[roomId]})
        for client in rooms[roomId]:
            try:
                await client.send_text(broadcast_data)
            except:
                pass
    return {'reports': room_reports[roomId]}