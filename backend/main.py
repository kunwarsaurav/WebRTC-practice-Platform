import logging
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
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("webrtc-backend")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
rooms: Dict[str, List[WebSocket]] = {}
client_user_map: Dict[WebSocket, str] = {}
room_reports: Dict[str, Dict[str, dict]] = {}

@app.on_event('startup')
async def startup_event():
    logger.info('Initializing heavy ML models...')
    init_models()
    logger.info('ML Models initialization complete.')

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
            for client in list(rooms.get(room_id, [])):
                if client != websocket:
                    try:
                        await client.send_text(data)
                    except Exception as e:
                        logger.error(f"Failed to send message to a client: {e}")
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
        logger.warning("Webhook URL not configured, skipped, skipping webhook.")
        return
    
    payload = {
        "roomId": room_id,
        "userId": user_id,
        "report": report
    }
    
    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"Sending webhook to {webhook_url} for user {user_id} in room {room_id}")
            response = await client.post(webhook_url, json=payload, timeout=10.0)
            response.raise_for_status()
            logger.info("Webhook sent successfully")
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}", exc_info=True)

def transcribe_with_google(audio_path: str):
    from google.cloud import speech
    from google.cloud import storage
    import soundfile as sf
    
    logger.info("Transcribing with Google Speech-to-Text...")
    bucket_name = os.environ.get("GCS_BUCKET_NAME")
    if not bucket_name or bucket_name == "your-gcs-bucket-name":
        raise ValueError("GCS_BUCKET_NAME not properly configured in .env")
    
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav.close()
    sf.write(temp_wav.name, y, sr, subtype='PCM_16')
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    blob_name = f"ielts_audio_{uuid.uuid4().hex}.wav"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(temp_wav.name)
    
    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    
    speech_client = speech.SpeechClient()
    
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        language_code="en-US",
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
        model="latest_long",
    )
    
    operation = speech_client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=600)
    
    blob.delete()
    try:
        os.unlink(temp_wav.name)
    except:
        pass
    
    transcript_text = ""
    segments = []
    word_timestamps = []
    
    for result in response.results:
        alternative = result.alternatives[0]
        transcript_text += alternative.transcript + " "
        
        if alternative.words:
            start_sec = alternative.words[0].start_time.total_seconds()
            end_sec = alternative.words[-1].end_time.total_seconds()
            
            segments.append({
                "start": start_sec,
                "end": end_sec,
                "text": alternative.transcript
            })
            
            for word_info in alternative.words:
                word_timestamps.append({
                    "word": word_info.word,
                    "start": word_info.start_time.total_seconds(),
                    "end": word_info.end_time.total_seconds()
                })
                
    return transcript_text.strip(), segments, word_timestamps

def process_audio_sync(temp_file_path: str, question: str):
    try:
        logger.info(f"process_audio_sync started for {temp_file_path}")
        use_google = os.environ.get("USE_GOOGLE_SPEECH", "false").lower() == "true"
        logger.info(f"USE_GOOGLE_SPEECH is set to: {use_google}")
        
        if use_google:
            logger.info("Initiating Google Speech-to-Text pipeline...")
            transcript_text, segments_list, word_timestamps_list = transcribe_with_google(temp_file_path)
            logger.info("Google STT complete. Transcript generated.")
        else:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key or api_key == "your_groq_api_key_here":
                logger.error("GROQ_API_KEY missing!")
                raise ValueError("GROQ_API_KEY is missing or invalid in .env")

            logger.info("Initiating Groq Whisper pipeline...")
            with open(temp_file_path, "rb") as audio_file:
                response = httpx.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data={
                        "model": "whisper-large-v3",
                        "response_format": "verbose_json",
                        "timestamp_granularities[]": ["word", "segment"]
                    },
                    files={"file": (os.path.basename(temp_file_path), audio_file, "audio/webm")},
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()

            transcript_text = result.get("text", "").strip()
            segments_list = result.get("segments", [])
            word_timestamps_list = result.get("words", [])
            logger.info("Groq Whisper complete. Transcript generated.")

        logger.info("Loading audio with librosa for feature extraction...")
        audio_array, sr = librosa.load(temp_file_path, sr=16000)
        
        logger.info("Passing data to evaluate_pipeline...")
        report = evaluate_pipeline(transcript=transcript_text, segments=segments_list, word_timestamps=word_timestamps_list, audio_path=temp_file_path, audio_array=audio_array, sample_rate=sr, lang='en-US', question=question)
        logger.info("evaluate_pipeline complete.")
        return report
    except Exception as e:
        logger.error("Pipeline Error processing audio", exc_info=True)
        return {'fluency': 0.0, 'lexical': 0.0, 'grammar': 0.0, 'pronunciation': 0.0, 'overall': 0.0, 'user_input': transcript_text if 'transcript_text' in locals() else 'N/A', 'feedback': {'fluency': f'Pipeline Error: {str(e)}', 'lexical': f'Pipeline Error: {str(e)}', 'grammar': f'Pipeline Error: {str(e)}', 'pronunciation': f'Pipeline Error: {str(e)}'}}

@app.post('/evaluate')
async def evaluate_audio(audio: UploadFile=File(...), roomId: str=Form(...), userId: str=Form(...), question: str=Form(None), background_tasks: BackgroundTasks=BackgroundTasks()):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f'{uuid.uuid4()}_{audio.filename}')
    with open(temp_file_path, 'wb') as buffer:
        buffer.write(await audio.read())
    file_size = os.path.getsize(temp_file_path)
    logger.info(f'Received audio from User {userId} in Room {roomId}, size:{file_size} bytes, content_type: {audio.content_type}')

    def cleanup():
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Cleaned up temp file: {temp_file_path}")
    background_tasks.add_task(cleanup)
    
    logger.info(f"[{roomId}:{userId}] Passing audio to thread for processing...")
    # Run the heavy AI models in a separate thread so we don't block other users
    report = await asyncio.to_thread(process_audio_sync, temp_file_path, question)
    logger.info(f"[{roomId}:{userId}] Processing thread returned report.")
    
    if roomId not in room_reports:
        room_reports[roomId] = {}
    room_reports[roomId][userId] = report
    
    logger.info(f"[{roomId}:{userId}] Triggering webhook...")
    # Trigger the webhook asynchronously
    background_tasks.add_task(send_webhook, roomId, userId, report)
    
    if roomId in rooms:
        logger.info(f"[{roomId}] Broadcasting evaluation results to {len(rooms[roomId])} clients...")
        broadcast_data = json.dumps({'type': 'evaluation-ready', 'reports': room_reports[roomId]})
        for client in rooms[roomId]:
            try:
                await client.send_text(broadcast_data)
            except Exception as e:
                logger.error(f"Failed to broadcast to a client in {roomId}: {e}")
    return {'reports': room_reports[roomId]}