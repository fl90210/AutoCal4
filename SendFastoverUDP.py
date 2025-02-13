import cv2
import numpy as np
import asyncio
import websockets
import socket
import struct
import json

# UDP & WebSocket Configurations
UDP_IP = "127.0.0.1"  # Change to receiver's IP if needed
UDP_PORT = 5005
WS_PORT = 8765

# Initialize UDP socket
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Initialize FAST detector
fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

# WebSocket Clients
clients = set()

# WebSocket server handler
async def websocket_handler(websocket, path):
    clients.add(websocket)
    try:
        async for message in websocket:
            pass  # No need to process messages from clients
    finally:
        clients.remove(websocket)

# Function to send data over WebSocket
async def send_to_websocket(data):
    if clients:
        message = json.dumps(data)
        await asyncio.gather(*[client.send(message) for client in clients])

# Function to process frames and send keypoints
async def process_frames():
    cap = cv2.VideoCapture(0)  # Use default webcam
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect FAST keypoints
        keypoints = fast.detect(gray, None)

        # Limit to 1000 keypoints for performance
        keypoints = keypoints[:1000] if len(keypoints) > 1000 else keypoints

        # Extract patches and prepare data
        patch_size = 16  # Box size around each keypoint
        keypoint_data = []
        udp_data = b''  # Binary buffer for UDP

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])

            # Extract a small patch
            x1, y1 = max(0, x - patch_size // 2), max(0, y - patch_size // 2)
            x2, y2 = min(frame.shape[1], x + patch_size // 2), min(frame.shape[0], y + patch_size // 2)
            patch = frame[y1:y2, x1:x2]

            # Resize to fixed size if necessary
            if patch.shape[:2] != (patch_size, patch_size):
                patch = cv2.resize(patch, (patch_size, patch_size))

            # Flatten patch and append coordinates (8-bit color)
            patch_bytes = patch.flatten().tobytes()
            udp_data += struct.pack("II", x, y) + patch_bytes

            # Add keypoint data for WebSocket
            keypoint_data.append({"x": x, "y": y, "patch": patch.tolist()})

        # Send UDP data in smaller chunks (MTU optimization)
        chunk_size = 4096  # UDP packet size
        for i in range(0, len(udp_data), chunk_size):
            udp_socket.sendto(udp_data[i:i+chunk_size], (UDP_IP, UDP_PORT))

        # Send WebSocket data
        await send_to_websocket(keypoint_data)

        # Draw keypoints for visualization
        frame_fast = cv2.drawKeypoints(frame, keypoints, None, color=(255, 0, 0))

        # Show the processed frame
        cv2.imshow("FAST Keypoints", frame_fast)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run both WebSocket and frame processing in parallel
async def main():
    websocket_server = websockets.serve(websocket_handler, "0.0.0.0", WS_PORT)
    await asyncio.gather(websocket_server, process_frames())

# Run event loop
asyncio.run(main())
