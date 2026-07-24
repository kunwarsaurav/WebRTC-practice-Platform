import React, { useState, useRef, useEffect } from 'react'
import ReactDOM from 'react-dom/client'
import { ArrowLeft, Clock, User, Mic, MicOff, PhoneOff, SkipForward, FileText } from 'lucide-react'
import './style.css'

const ICE_SERVERS = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' }
  ]
};

const TOPICS = [
  "Describe a place in your country that you would like to recommend to visitors. You should say: where it is, what people can see and do there, and explain why you would recommend this place.",
  "Describe a book you have recently read. You should say: what kind of book it is, what it is about, what sort of people would enjoy it, and explain why you liked it.",
  "Describe a time you taught something new to a younger person. You should say: who you taught, what you taught, how you felt about it, and explain why you decided to teach them.",
  "Describe a skill you learned when you were a child. You should say: what the skill was, who taught it to you, how you learned it, and explain why this skill is important to you.",
  "Describe an interesting conversation you had with a stranger. You should say: where you met them, what you talked about, why you found it interesting, and explain how it affected you."
];

const SESSION_DURATION = 600;

function App() {
  const [roomId, setRoomId] = useState('');
  const [status, setStatus] = useState('Disconnected');
  const statusRef = useRef('Disconnected');
  useEffect(() => { statusRef.current = status; }, [status]);
  const [isMuted, setIsMuted] = useState(false);
  const [isPartnerMuted, setIsPartnerMuted] = useState(false);
  const [topicIndex, setTopicIndex] = useState(0);
  const [secondsRemaining, setSecondsRemaining] = useState(SESSION_DURATION);
  const [reports, setReports] = useState(null);

  const [isRecording, setIsRecording] = useState(false);
  const [hasRecorded, setHasRecorded] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const recordedBlobRef = useRef(null);

  const localStreamRef = useRef(null);
  const peerConnectionRef = useRef(null);
  const wsRef = useRef(null);
  const remoteAudioRef = useRef(null);
  const timerRef = useRef(null);
  const pendingCandidatesRef = useRef([]);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const isProcessingRef = useRef(false);
  const userIdRef = useRef(Math.random().toString(36).substring(7));

  useEffect(() => {
    if (status === 'Connected') {
      timerRef.current = setInterval(() => {
        setSecondsRemaining(prev => {
          if (prev <= 1) {
            clearInterval(timerRef.current);
            handleLeave(true);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    } else {
      clearInterval(timerRef.current);
      if (status === 'Disconnected') {
        setSecondsRemaining(SESSION_DURATION);
        setIsRecording(false);
        setHasRecorded(false);
        setIsSubmitted(false);
        recordedBlobRef.current = null;
      }
    }
    return () => clearInterval(timerRef.current);
  }, [status]);

  const formatTime = (totalSeconds) => {
    const m = Math.floor(totalSeconds / 60).toString().padStart(2, '0');
    const s = (totalSeconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  const uploadAudio = async (blob) => {
    console.log(`[Submission] Preparing to upload audio. Blob size: ${blob.size} bytes`);
    const formData = new FormData();
    formData.append("audio", blob, "recording.webm");
    formData.append("roomId", roomId);
    formData.append("userId", userIdRef.current);
    formData.append("question", TOPICS[topicIndex]);

    const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'localhost:8000';
    const httpUrl = BACKEND_URL.includes('localhost') ? `http://${BACKEND_URL}` : `https://${BACKEND_URL}`;
    console.log(`[Submission] Sending POST request to: ${httpUrl}/evaluate`);

    try {
      const response = await fetch(`${httpUrl}/evaluate`, {
        method: "POST",
        body: formData,
        headers: {
          "Bypass-Tunnel-Reminder": "true"
        }
      });
      const data = await response.json();
      console.log("[Submission] Evaluation successfully initiated. Server Response:", data);
    } catch (e) {
      console.error("[Submission Error] Failed to upload audio:", e);
      alert("Evaluation failed.");
      setIsSubmitted(false);
    }
  };

  const connectToSignalingServer = async (roomId) => {
    const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'localhost:8000';
    const isHttps = window.location.protocol === 'https:' || BACKEND_URL.startsWith('https://') || BACKEND_URL.startsWith('wss://');
    const protocol = isHttps && !BACKEND_URL.includes('localhost') ? 'wss' : 'ws';
    const cleanUrl = BACKEND_URL.replace(/^https?:\/\//, '').replace(/^wss?:\/\//, '');
    const wsUrl = `${protocol}://${cleanUrl}/ws/${roomId}?userId=${userIdRef.current}`;

    console.log(`[WebSocket] Attempting to connect to signaling server at: ${wsUrl}`);
    wsRef.current = new WebSocket(wsUrl);
    
    let pingInterval;
    wsRef.current.onopen = () => {
      console.log("[WebSocket] Connection established successfully!");
      setStatus('Waiting');
      pingInterval = setInterval(() => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'ping' }));
        }
      }, 30000);
    };

    wsRef.current.onerror = (error) => {
      console.error("[WebSocket Error] Connection error occurred:", error);
    };

    wsRef.current.onmessage = async (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'ping') return;
      if (message.type === 'user-joined') createOffer();
      else if (message.type === 'offer') await handleOffer(message.sdp);
      else if (message.type === 'answer') await handleAnswer(message.sdp);
      else if (message.type === 'ice-candidate') await handleNewICECandidateMsg(message.candidate);
      else if (message.type === 'user-left') {
        if (statusRef.current === 'Connected') {
          handleLeave(true);
        }
      }
      else if (message.type === 'end-call') {
        if (statusRef.current === 'Connected' || statusRef.current === 'Waiting') {
          handleLeave(true);
        }
      }
      else if (message.type === 'mute-status') {
        setIsPartnerMuted(message.isMuted);
      }
      else if (message.type === 'evaluation-ready') {
        setReports(message.reports);
        if (statusRef.current === 'Processing' && Object.keys(message.reports).length > 0) {
          setStatus('ReportReady');
        }
      }
    };
    wsRef.current.onclose = () => {
      clearInterval(pingInterval);
      if (statusRef.current === 'Connected' || statusRef.current === 'Waiting') {
        setStatus('Disconnected');
      }
    };
  };

  const initializePeerConnection = () => {
    const pc = new RTCPeerConnection(ICE_SERVERS);
    peerConnectionRef.current = pc;
    pendingCandidatesRef.current = [];
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach(track => pc.addTrack(track, localStreamRef.current));
    }
    pc.onicecandidate = (event) => {
      if (event.candidate && wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ice-candidate', candidate: event.candidate }));
      }
    };
    pc.oniceconnectionstatechange = () => {
      if (pc.iceConnectionState === 'connected') setStatus('Connected');
      else if (pc.iceConnectionState === 'disconnected' || pc.iceConnectionState === 'failed' || pc.iceConnectionState === 'closed') {
        if (!isProcessingRef.current) {
          if (statusRef.current === 'Connected') {
            handleLeave(true);
          } else {
            setStatus('Disconnected');
          }
        }
      }
    };
    pc.ontrack = (event) => {
      if (remoteAudioRef.current && event.streams[0]) {
        remoteAudioRef.current.srcObject = event.streams[0];
      }
    };
  };

  const createOffer = async () => {
    if (!peerConnectionRef.current) return;
    const offer = await peerConnectionRef.current.createOffer();
    await peerConnectionRef.current.setLocalDescription(offer);
    wsRef.current.send(JSON.stringify({ type: 'offer', sdp: peerConnectionRef.current.localDescription }));
  };

  const handleOffer = async (sdp) => {
    if (!peerConnectionRef.current) return;
    await peerConnectionRef.current.setRemoteDescription(new RTCSessionDescription(sdp));
    const answer = await peerConnectionRef.current.createAnswer();
    await peerConnectionRef.current.setLocalDescription(answer);
    wsRef.current.send(JSON.stringify({ type: 'answer', sdp: peerConnectionRef.current.localDescription }));
    
    for (const candidate of pendingCandidatesRef.current) {
      try { await peerConnectionRef.current.addIceCandidate(new RTCIceCandidate(candidate)); }
      catch (e) { console.error(e); }
    }
    pendingCandidatesRef.current = [];
  };

  const handleAnswer = async (sdp) => {
    if (!peerConnectionRef.current) return;
    await peerConnectionRef.current.setRemoteDescription(new RTCSessionDescription(sdp));
    
    for (const candidate of pendingCandidatesRef.current) {
      try { await peerConnectionRef.current.addIceCandidate(new RTCIceCandidate(candidate)); }
      catch (e) { console.error(e); }
    }
    pendingCandidatesRef.current = [];
  };

  const handleNewICECandidateMsg = async (candidate) => {
    if (!peerConnectionRef.current) return;
    try { 
      if (!peerConnectionRef.current.remoteDescription) {
        pendingCandidatesRef.current.push(candidate);
      } else {
        await peerConnectionRef.current.addIceCandidate(new RTCIceCandidate(candidate)); 
      }
    }
    catch (e) { console.error(e); }
  };

  const startRecording = () => {
    if (!mediaRecorderRef.current) return;
    console.log("[Recording] Starting audio capture...");
    audioChunksRef.current = [];
    mediaRecorderRef.current.start(1000);
    setIsRecording(true);
    setHasRecorded(false);
  };

  const stopRecording = () => {
    if (!mediaRecorderRef.current) return;
    console.log("[Recording] Stopping audio capture...");
    mediaRecorderRef.current.stop();
    setIsRecording(false);
    setHasRecorded(true);
  };

  const handleSubmit = async () => {
    if (!recordedBlobRef.current) return alert("Please record your answer first.");
    setIsSubmitted(true);
    await uploadAudio(recordedBlobRef.current);
  };

  const handleJoin = async () => {
    if (!roomId) return alert("Please enter a room ID");
    try {
      console.log("[Setup] Requesting microphone access...");
      const stream = await navigator.mediaDevices.getUserMedia
        ({
          audio: {
            echoCancellation: true,  // Enable echo cancellation
            noiseSuppression: true,    // Enable noise suppression
            autoGainControl: false,    // Enable auto gain control
            sampleRate: 48000,
            channelCount: 1
          },
          video: false
        });
      console.log("[Setup] Microphone access granted successfully.");
      localStreamRef.current = stream;

      const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4';
      console.log(`[Setup] Initializing MediaRecorder with MIME type: ${mimeType}`);
      const mr = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mr;

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mr.onstop = () => {
        recordedBlobRef.current = new Blob(audioChunksRef.current, { type: mimeType });
        console.log(`[Recording] Blob created successfully. Size: ${recordedBlobRef.current.size} bytes`);
      };

      initializePeerConnection();
      connectToSignalingServer(roomId);
    } catch (err) {
      console.error("[Setup Error] Error joining session:", err);
      if (err.name === 'NotAllowedError' || err.name === 'NotFoundError') {
        alert("Microphone access denied or no microphone found.");
      } else {
        alert("An error occurred while setting up the session. Check console for details.");
      }
    }
  };

  const handleLeave = async (process = false) => {

    if (process && !hasRecorded && !isRecording){
      process = false;
    }
    if (process && (!reports || Object.keys(reports).length === 0)) {
      isProcessingRef.current = true;
      setStatus('Processing');
      if (!isSubmitted && recordedBlobRef.current) {
        setIsSubmitted(true);
        await uploadAudio(recordedBlobRef.current);
      }
    } else if (reports && Object.keys(reports).length > 0) {
      isProcessingRef.current = false;
      setStatus('ReportReady');
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') mediaRecorderRef.current.stop();
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) wsRef.current.send(JSON.stringify({ type: 'end-call' }));
    } else {
      isProcessingRef.current = false;
      setStatus('Disconnected');
      setReports(null);
      setTopicIndex(0);
      setSecondsRemaining(SESSION_DURATION);
      setIsRecording(false);
      setHasRecorded(false);
      setIsSubmitted(false);
      recordedBlobRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }

    if (wsRef.current) {
      if (process && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'end-call' }));
      } else {
        wsRef.current.close();
        wsRef.current = null;
      }
    }

    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
      peerConnectionRef.current = null;
    }
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach(t => t.stop());
      localStreamRef.current = null;
    }
  };

  const toggleMute = () => {
    if (localStreamRef.current) {
      const track = localStreamRef.current.getAudioTracks()[0];
      if (track) { 
        track.enabled = !track.enabled; 
        setIsMuted(!track.enabled);
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'mute-status', isMuted: !track.enabled }));
        }
      }
    }
  };

  useEffect(() => {
    return () => handleLeave(false);
  }, []);

  if (status === 'Disconnected') {
    return (
      <div className="join-screen glass-panel">
        <h1 style={{ marginBottom: '30px', color: 'var(--primary)' }}>IELTS Practice</h1>
        <input className="join-input" type="text" placeholder="Enter Room ID" value={roomId} onChange={(e) => setRoomId(e.target.value)} />
        <button className="join-btn" onClick={handleJoin}>Join Practice Session</button>
      </div>
    );
  }

  if (status === 'Processing') {
    return (
      <div className="join-screen glass-panel">
        <h2 style={{ color: 'var(--primary)' }}>Session Complete!</h2>
        <p style={{ color: 'var(--text-muted)', marginTop: '10px' }}>Processing your conversation with AI...</p>
        <div style={{ marginTop: '20px' }}>
          <Clock className="animate-spin" size={32} color="var(--primary)" style={{ margin: '0 auto' }} />
        </div>
      </div>
    );
  }

  if (status === 'ReportReady') {
    return (
      <div className="app-container">
        <div className="header">
          <button className="back-btn" onClick={() => {
            setStatus('Disconnected');
            setReports(null);
            setTopicIndex(0);
            setSecondsRemaining(SESSION_DURATION);
            setIsRecording(false);
            setHasRecorded(false);
            setIsSubmitted(false);
            recordedBlobRef.current = null;
            if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
          }}>
            <ArrowLeft size={20} />
          </button>
          <div className="title">Evaluation Reports</div>
          <div></div>
        </div>

        <div className="reports-container">
          {reports && Object.entries(reports).map(([uid, report]) => (
            <div key={uid} className="report-card glass-panel">
              <h3>
                <FileText size={24} />
                {uid === userIdRef.current ? "Your Report" : "Partner's Report"}
              </h3>
              <div className={`report-overall ${report.overall >= 7 ? 'score-green' : report.overall >= 5.5 ? 'score-orange' : 'score-red'}`}>
                Band {report.overall}
              </div>
              <div className="report-metrics">
                <div className="metric-row"><span>Fluency:</span> <strong>{report.fluency}</strong></div>
                <div className="metric-row"><span>Lexical Resource:</span> <strong>{report.lexical}</strong></div>
                <div className="metric-row"><span>Grammar:</span> <strong>{report.grammar}</strong></div>
                <div className="metric-row"><span>Pronunciation:</span> <strong>{report.pronunciation}</strong></div>
              </div>

              {report.user_input && (
                <div className="transcript-box">
                  <strong style={{ display: 'block', marginBottom: '6px' }}>Your Transcript</strong>
                  <span style={{ fontStyle: 'italic' }}>"{report.user_input}"</span>
                </div>
              )}
              {report.feedback && typeof report.feedback === 'object' ? (
                <div className="feedback-section">
                  {Object.entries(report.feedback).map(([category, text]) => (
                    <div key={category} className="feedback-box">
                      <strong className="feedback-title">{category} Feedback</strong>
                      {text}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="feedback-box">
                  {report.feedback}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <div className="header">
        <button className="back-btn" onClick={() => handleLeave(true)}><ArrowLeft size={20} /></button>
        <div className="title">Practice Session</div>
        <div className="timer"><Clock size={18} /> <span>{formatTime(secondsRemaining)}</span></div>
      </div>

      <div className="avatars-section">
        <div className="avatar-wrapper">
          <div className="avatar you"><User size={28} /></div>
          <div className="avatar-label">You</div>
        </div>
        <div className="avatar-line"></div>
        <div className="avatar-wrapper">
          <div className={`avatar partner ${status === 'Waiting' ? 'opacity-50' : ''}`}><User size={28} /></div>
          <div className="avatar-label">Partner</div>
        </div>
      </div>

      {reports && Object.keys(reports).length > 0 ? (
        <div className="reports-container">
          {Object.entries(reports).map(([uid, report]) => (
            <div key={uid} className="report-card glass-panel">
              <h3>
                <FileText size={24} />
                {uid === userIdRef.current ? "Your Report" : "Partner's Report"}
              </h3>
              <div className={`report-overall ${report.overall >= 7 ? 'score-green' : report.overall >= 5.5 ? 'score-orange' : 'score-red'}`}>
                Band {report.overall}
              </div>
              <div className="report-metrics">
                <div className="metric-row"><span>Fluency:</span> <strong>{report.fluency}</strong></div>
                <div className="metric-row"><span>Lexical Resource:</span> <strong>{report.lexical}</strong></div>
                <div className="metric-row"><span>Grammar:</span> <strong>{report.grammar}</strong></div>
                <div className="metric-row"><span>Pronunciation:</span> <strong>{report.pronunciation}</strong></div>
              </div>

              {report.user_input && (
                <div className="transcript-box">
                  <strong style={{ display: 'block', marginBottom: '6px' }}>Your Transcript</strong>
                  <span style={{ fontStyle: 'italic' }}>"{report.user_input}"</span>
                </div>
              )}
              {report.feedback && typeof report.feedback === 'object' ? (
                <div className="feedback-section">
                  {Object.entries(report.feedback).map(([category, text]) => (
                    <div key={category} className="feedback-box">
                      <strong className="feedback-title">{category} Feedback</strong>
                      {text}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="feedback-box">
                  {report.feedback}
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="topic-card glass-panel">
          <div className="topic-header">CURRENT TOPIC</div>
          <div className="topic-content">{TOPICS[topicIndex]}</div>
          <div className="topic-footer">
            <span>Topic {topicIndex + 1} of {TOPICS.length}</span>
            <div className="dots">
              {TOPICS.map((_, i) => <div key={i} className={`dot ${i === topicIndex ? 'active' : ''}`}></div>)}
            </div>
          </div>
          <button className="next-topic-btn" onClick={() => setTopicIndex(p => (p + 1) % TOPICS.length)}>
            Next <span style={{ fontSize: '18px', marginLeft: '2px', lineHeight: 1 }}>›</span>
          </button>
        </div>
      )}

      <div className="visualizer-section">
        <div className="visualizer-wrapper">
          <div className="bars">
            {!isMuted ? <><div className="bar"></div><div className="bar"></div><div className="bar"></div><div className="bar"></div></> : <div className="bar-inactive"></div>}
          </div>
          <div className="avatar-label">You</div>
        </div>
        <div className="visualizer-wrapper">
          <div className="bars">
            {status === 'Connected' && !isPartnerMuted ? <><div className="bar partner-bar"></div><div className="bar partner-bar"></div><div className="bar partner-bar"></div><div className="bar partner-bar"></div></> : <div className="partner-inactive-dashed"></div>}
          </div>
          <div className="avatar-label">Partner {status === 'Waiting' && '(Waiting)'}</div>
        </div>
      </div>

      <div className="controls-section">
        {status === 'Connected' && !isSubmitted && (
          <>
            {!isRecording ? (
              <button className="btn-action btn-record" onClick={startRecording}>
                Record Answer
              </button>
            ) : (
              <button className="btn-action btn-stop" onClick={stopRecording}>
                Stop Recording
              </button>
            )}
          </>
        )}

        {hasRecorded && !isRecording && !isSubmitted && status === 'Connected' && (
          <button className="btn-action btn-submit" onClick={handleSubmit}>
            Submit Answer
          </button>
        )}

        {isSubmitted && status === 'Connected' && (!reports || !reports[userIdRef.current]) && (
          <div style={{ padding: '10px 20px', color: '#10b981', fontWeight: 'bold', background: '#ecfdf5', borderRadius: '8px' }}>
            AI is evaluating... You can continue talking with your partner!
          </div>
        )}

        <button className={`control-btn ${isMuted ? 'muted' : ''}`} onClick={toggleMute}>
          {isMuted ? <MicOff size={24} /> : <Mic size={24} />}
        </button>
        <button className="control-btn end-call" onClick={() => handleLeave(true)}><PhoneOff size={28} /></button>
        <button className="control-btn" onClick={() => setTopicIndex(p => (p + 1) % TOPICS.length)}><SkipForward size={24} /></button>
      </div>
      <audio ref={remoteAudioRef} autoPlay />
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('newfeature-root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)