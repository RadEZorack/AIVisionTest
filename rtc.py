import cv2
import numpy as np
import asyncio
import json
from fastapi import APIRouter, WebSocket
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceCandidate
from av import VideoFrame

router = APIRouter()

class VideoProcessor(VideoStreamTrack):
    """
    WebRTC Video Processor that receives frames, computes motion diffs, and returns minimal data.
    """

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.previous_frame = None

    async def recv(self):
        """ Receive and process frames from WebRTC stream. """
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # Resize and convert to grayscale for faster processing
        img = cv2.resize(img, (320, 240))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        motion_data = None
        if self.previous_frame is not None:
            diff = cv2.absdiff(self.previous_frame, gray)
            motion_data = np.sum(diff)
            print(f"🖥️ Motion Detected: {motion_data}", flush=True)

        self.previous_frame = gray

        # Return the processed frame (optional, mostly for debugging)
        return VideoFrame.from_ndarray(gray, format="gray")

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
