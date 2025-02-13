import socket
import struct

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while True:
    data, addr = sock.recvfrom(4096)  # Adjust buffer size as needed
    x, y = struct.unpack("II", data[:8])  # Use "II" for unsigned 32-bit integers
    print(f"Keypoint at: ({x}, {y})")
