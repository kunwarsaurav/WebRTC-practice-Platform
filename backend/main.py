import os
import tempfile
import uuid
from typing import Dict, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import json
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from scorer import init_models, evaluate_pipeline
load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
rooms: Dict[str, List[WebSocket]] = {}
client_user_map: Dict[WebSocket, str] = {}
room_reports: Dict[str, Dict[str, dict]] = {}
whisper_model = WhisperModel('large-v3', device='cuda', compute_type='float16')
print('Whisper model loaded.')

@app.on_event('startup')
async def startup_event():
    print('Initializing heavy ML models...')
    init_models()
    print('ML Models initialization complete.')

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
    try:
        segments_gen, info = whisper_model.transcribe(temp_file_path, beam_size=5, word_timestamps=True)
        segments_list = []
        word_timestamps_list = []
        transcript_text = ''
        for segment in segments_gen:
            transcript_text += segment.text + ' '
            segments_list.append({'start': segment.start, 'end': segment.end, 'text': segment.text})
            if segment.words:
                for word in segment.words:
                    word_timestamps_list.append({'word': word.word, 'start': word.start, 'end': word.end})
        transcript_text = transcript_text.strip()
        audio_array = decode_audio(temp_file_path, sampling_rate=16000)
        sr = 16000
        report = evaluate_pipeline(transcript=transcript_text, segments=segments_list, word_timestamps=word_timestamps_list, audio_path=temp_file_path, audio_array=audio_array, sample_rate=sr, lang='en-US', question=question)
    except Exception as e:
        import traceback
        traceback.print_exc()
        report = {'fluency': 0.0, 'lexical': 0.0, 'grammar': 0.0, 'pronunciation': 0.0, 'overall': 0.0, 'user_input': transcript_text if 'transcript_text' in locals() else 'N/A', 'feedback': {'fluency': f'Pipeline Error: {str(e)}', 'lexical': f'Pipeline Error: {str(e)}', 'grammar': f'Pipeline Error: {str(e)}', 'pronunciation': f'Pipeline Error: {str(e)}'}}
    if roomId not in room_reports:
        room_reports[roomId] = {}
    room_reports[roomId][userId] = report
    if roomId in rooms:
        broadcast_data = json.dumps({'type': 'evaluation-ready', 'reports': room_reports[roomId]})
        for client in rooms[roomId]:
            try:
                await client.send_text(broadcast_data)
            except:
                pass
    return {'reports': room_reports[roomId]}