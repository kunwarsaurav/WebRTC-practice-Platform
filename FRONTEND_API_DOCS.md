# AI Evaluation Engine - Frontend Integration Guide

This document outlines how the Frontend should connect to the AI Evaluation Engine (hosted at `ml.synthbitietlsboost.com`). 

The AI Engine handles **WebRTC Signaling** (so users can talk peer-to-peer) and **AI Scoring** (to evaluate the audio).

---

## 1. Room Creation (REST)

Before connecting to the WebSocket, you must request a unique Room ID from the AI Engine.

* **URL:** `https://ml.synthbitietlsboost.com/create-room`
* **Method:** `POST`

### Response
```json
{
  "roomId": "a1b2c3d4-5678-90ef-ghij-klmnopqrstuv"
}
```

---

## 2. WebRTC Signaling (WebSocket)

The Frontend must connect directly to the AI Engine's WebSocket to establish the audio call between the two users.

* **URL:** `wss://ml.synthbitietlsboost.com/ws/{roomId}?userId={userId}`
* **Protocol:** `WebSocket`

### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `roomId` | `Path` | The unique ID of the room (provided by the AI Engine via `/create-room`). |
| `userId` | `Query` | The unique ID of the user connecting (provided by the Main Backend). |

### WebSocket Events (JSON)
Once connected, the Frontend should listen for and send the following JSON stringified messages:

**Incoming Messages (Listen for these):**
* `{"type": "user-joined"}`: Fired when the partner joins. The Frontend should create a WebRTC Offer.
* `{"type": "offer", "sdp": ...}`: Receive a WebRTC offer from the partner.
* `{"type": "answer", "sdp": ...}`: Receive a WebRTC answer from the partner.
* `{"type": "ice-candidate", "candidate": ...}`: Receive ICE candidates to establish the connection.
* `{"type": "user-left"}`: Fired if the partner disconnects.
* `{"type": "evaluation-ready", "reports": {...}}`: Fired when the AI has finished processing the final score.

**Outgoing Messages (Send these):**
* `{"type": "offer", "sdp": ...}`: Send local WebRTC offer.
* `{"type": "answer", "sdp": ...}`: Send local WebRTC answer.
* `{"type": "ice-candidate", "candidate": ...}`: Send local ICE candidates.

---

## 2. Audio Evaluation Upload (REST)

When the practice session concludes, the Frontend must upload the locally recorded audio file to the AI Engine for evaluation.

* **URL:** `https://ml.synthbitietlsboost.com/evaluate`
* **Method:** `POST`
* **Content-Type:** `multipart/form-data`

### Form Data Fields
| Field | Type | Description | Required |
| :--- | :--- | :--- | :--- |
| `audio` | `File (Blob)` | The `.webm` audio recording from the user's microphone. | Yes |
| `roomId` | `String` | The ID of the room the user was in. | Yes |
| `userId` | `String` | The unique ID of the user. | Yes |
| `question` | `String` | The exact text of the IELTS question they were discussing. | No (Recommended) |

### Response
The endpoint will return a JSON object containing the evaluation reports for the room.

```json
{
  "reports": {
    "usr_987654321": {
      "fluency": 7.5,
      "lexical": 8.0,
      "grammar": 6.5,
      "pronunciation": 7.0,
      "overall": 7.5,
      "feedback": {
         "fluency": "Excellent job! You speak very naturally...",
         "lexical": "Great vocabulary...",
         ...
      }
    }
  }
}
```

---

## 3. Example Frontend Implementation (JavaScript)

```javascript
// 1. Connection Variables 
const roomId = "room_456";
const userId = "usr_987654321";

// 2. Establish WebRTC Signaling Connection
const wsUrl = `wss://ml.synthbitietlsboost.com/ws/${roomId}?userId=${userId}`;
const socket = new WebSocket(wsUrl);

socket.onopen = () => {
    console.log("Connected to AI Signaling Server!");
};

socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    if (message.type === 'user-joined') {
        // Create WebRTC Offer
    } else if (message.type === 'offer') {
        // Handle Offer, Create Answer
    } else if (message.type === 'answer') {
        // Handle Answer
    } else if (message.type === 'ice-candidate') {
        // Add ICE candidate
    } else if (message.type === 'evaluation-ready') {
        console.log("AI Report Received via WebSocket:", message.reports);
    }
};

// 3. Upload Audio for AI Scoring (When session ends)
async function uploadAudioToAI(audioBlob, questionText) {
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");
    formData.append("roomId", roomId);
    formData.append("userId", userId);
    formData.append("question", questionText);

    try {
        const response = await fetch("https://ml.synthbitietlsboost.com/evaluate", {
            method: "POST",
            body: formData
        });
        
        const aiReport = await response.json();
        console.log("AI Report Received via HTTP Response:", aiReport);
    } catch (error) {
        console.error("Failed to upload audio:", error);
    }
}
```


// This is standard, built-in JavaScript that works in any browser
const socket = new WebSocket("wss://ml.synthbitietlsboost.com/ws/room_456?userId=123");