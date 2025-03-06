from openai import OpenAI
import cv2
import numpy as np
import asyncio
import json
import base64
import io
import os
from fastapi import APIRouter, WebSocket
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceCandidate
from av import VideoFrame
from PIL import Image

router = APIRouter()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)

class VideoProcessor(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track
        self.previous_frame = None
        self.initial_frame_sent = False

    async def send_to_openai(self, img):
        """ Send image data to OpenAI API """
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_type="image/jpeg"
        img_b64_str = base64.b64encode(buffered.getvalue()).decode()

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI with real-time vision analysis capabilities."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
                        },
                    ],
                }
            ],
            stream=True,
        )

        for chunk in completion:
            print(chunk.choices[0].delta.content or "", end="")

        return completion.choices[0].message.content

    async def recv(self):
        """ Capture and process frames """
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        img = cv2.resize(img, (320, 240))
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not self.initial_frame_sent:
            self.initial_frame_sent = True
            print("📸 Initial Frame Sent to AI", flush=True)
            ai_response = await self.send_to_openai(pil_img)
            print(f"🤖 AI Initial Analysis: {ai_response}", flush=True)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(self.previous_frame, gray) if self.previous_frame is not None else None
            self.previous_frame = gray

            if diff is not None and np.sum(diff) > 5000:  # Only send if change is significant
                diff_img = Image.fromarray(diff)
                ai_response = await self.send_to_openai(diff_img)
                print(f"🤖 AI Response to Diff: {ai_response}", flush=True)

        return frame

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    pc = RTCPeerConnection()

    @pc.on("icecandidate")
    async def on_ice_candidate(candidate):
        """Send ICE candidates to the frontend"""
        if candidate:
            print("❄️ ICE Candidate Sent to Client:", candidate, flush=True)
            await websocket.send_text(json.dumps({"candidate": candidate.to_sdp()}))

    @pc.on("iceconnectionstatechange")
    def on_ice_connection_state_change():
        print(f"🔄 ICE Connection State: {pc.iceConnectionState}", flush=True)

    @pc.on("track")
    def on_track(track):
        print(f"🔍 Track received: {track.kind}", flush=True)

        if track.kind == "video":
            processor = VideoProcessor(track)

            async def process():
                while True:
                    await processor.recv()  # Process frames
                    await asyncio.sleep(0)

            asyncio.create_task(process())

    # Receive WebRTC Offer
    offer = await websocket.receive_text()
    offer = json.loads(offer)
    rtc_offer = RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
    await pc.setRemoteDescription(rtc_offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    response = json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    await websocket.send_text(response)

    # ✅ Corrected: Parse ICE candidate manually before adding it to PeerConnection
    while True:
        message = await websocket.receive_text()
        message = json.loads(message)

        if "candidate" in message:
            print("📡 ICE Candidate Received from Client:", message["candidate"], flush=True)

            # Parse ICE candidate
            candidate_parts = message["candidate"].split()
            if len(candidate_parts) >= 8:
                candidate = RTCIceCandidate(
                    component=int(candidate_parts[1]),
                    foundation=candidate_parts[0],
                    ip=candidate_parts[4],
                    port=int(candidate_parts[5]),
                    priority=int(candidate_parts[3]),
                    protocol=candidate_parts[2],
                    type=candidate_parts[7],
                    relatedAddress=None,
                    relatedPort=None,
                    sdpMid="0",  # Required field
                    sdpMLineIndex=0,  # Required field
                )
                await pc.addIceCandidate(candidate)

        await asyncio.sleep(1)  # Keep loop running
