import cv2
import numpy as np
import time  # High-precision timing
from scipy.spatial import cKDTree

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create FAST and ORB detectors
fast = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
orb = cv2.ORB_create(nfeatures=1000)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Measure FAST Keypoint Extraction Time ---
    start_time_fast = time.perf_counter()
    fast_keypoints = fast.detect(gray, None)
    elapsed_time_fast = time.perf_counter() - start_time_fast

    # --- Limit FAST keypoints to prevent overload ---
    fast_keypoints = fast_keypoints[:2000] if len(fast_keypoints) > 2000 else fast_keypoints

    # --- Measure ORB Keypoint Extraction Time ---
    start_time_orb = time.perf_counter()
    orb_keypoints = orb.detect(gray, None)
    elapsed_time_orb = time.perf_counter() - start_time_orb

    # Convert keypoints to numpy arrays for KD-Tree matching
    fast_points = np.array([kp.pt for kp in fast_keypoints])
    orb_points = np.array([kp.pt for kp in orb_keypoints])

    # --- Measure Time for Finding Common Keypoints ---
    start_time_match = time.perf_counter()
    common_keypoints = []
    if len(fast_points) > 0 and len(orb_points) > 0:
        tree = cKDTree(fast_points)
        distances, indices = tree.query(orb_points, distance_upper_bound=1.0)  # Match within 1.0 pixel

        # Filter valid matches
        valid_matches = distances < 1.0
        common_keypoints = [fast_keypoints[i] for i in indices[valid_matches]]
    
    # Limit the number of common keypoints
    common_keypoints = common_keypoints[:500] if len(common_keypoints) > 500 else common_keypoints
    elapsed_time_match = time.perf_counter() - start_time_match

    # Draw FAST keypoints (Blue)
    frame_fast = frame.copy()
    cv2.drawKeypoints(frame, fast_keypoints, frame_fast, color=(255, 0, 0), flags=0)

    # Draw ORB keypoints (Green)
    frame_orb = frame.copy()
    cv2.drawKeypoints(frame, orb_keypoints, frame_orb, color=(0, 255, 0), flags=0)

    # Draw Common keypoints (Red)
    frame_common = frame.copy()
    cv2.drawKeypoints(frame, common_keypoints, frame_common, color=(0, 0, 255), flags=0)

    # Display windows
    cv2.imshow("FAST Keypoints (Blue)", frame_fast)
    cv2.imshow("ORB Keypoints (Green)", frame_orb)
    cv2.imshow("Common Keypoints (Red)", frame_common)

    # Print statistics
    print(f"FAST: {len(fast_keypoints)} | ORB: {len(orb_keypoints)} | Common: {len(common_keypoints)}")
    print(f"Time: FAST = {elapsed_time_fast:.6f}s | ORB = {elapsed_time_orb:.6f}s | Match = {elapsed_time_match:.6f}s")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
