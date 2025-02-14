import cv2
import numpy as np
import time  # Import time module

# Initialize the webcam
cap = cv2.VideoCapture(1)  # Change if using an external webcam

# Create a FAST detector
fast = cv2.FastFeatureDetector_create(threshold=35, nonmaxSuppression=True)
orb = cv2.ORB_create(nfeatures=1000)

# Define a Patch object to store image data and keypoints
class Patch:
    def __init__(self, image, position, patch_size):
        self.image = image  # The cropped patch image
        self.position = position  # (x, y) position in the original frame
        self.keypoints = []  # FAST keypoints detected inside this patch
        self.patch_size = patch_size  # Patch size

    def detect_keypoints(self, fast_detector):
        """Detect FAST keypoints inside the patch and store them."""
        gray_patch = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.keypoints = fast_detector.detect(gray_patch)  # Detect keypoints in the patch

    def map_keypoints_to_global(self):
        """Convert patch-local keypoints to global coordinates."""
        global_keypoints = []
        for kp in self.keypoints:
            # Map local patch keypoints to global frame coordinates
            global_x = int(kp.pt[0]) + self.position[0]
            global_y = int(kp.pt[1]) + self.position[1]
            global_keypoints.append((global_x, global_y))
        return global_keypoints


while True:
    # Capture frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Measure FAST keypoint extraction time ---
    start_time = time.perf_counter()  # Start the timer
    keypoints = fast.detect(gray, None)  # FAST keypoint detection
    elapsed_time = time.perf_counter() - start_time  # Calculate elapsed time
    print(f"FAST Keypoint Extraction Time: {elapsed_time:.6f} seconds")  # Print result
    # -------------------------------------------------

    # Sort keypoints in a stable way (top-left to bottom-right)
    keypoints = sorted(keypoints, key=lambda kp: (kp.pt[1], kp.pt[0]))

    # Draw keypoints on the original frame
    frame_with_keypoints = frame.copy()
    cv2.drawKeypoints(frame, keypoints, frame_with_keypoints, color=(0, 255, 0), flags=0)

    # Extract patches and store them as objects
    patch_size = 8  # Size of each extracted patch
    patches = []  # List of Patch objects
    motion_arrows = []  # Store arrows from original keypoints to detected keypoints

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        x1, y1 = max(0, x - patch_size // 2), max(0, y - patch_size // 2)
        x2, y2 = min(frame.shape[1], x + patch_size // 2), min(frame.shape[0], y + patch_size // 2)

        patch_img = frame[y1:y2, x1:x2]  # Crop patch
        if patch_img.shape[0] > 0 and patch_img.shape[1] > 0:  # Ensure valid patch
            resized_patch = cv2.resize(patch_img, (patch_size, patch_size))
            patch = Patch(resized_patch, (x1, y1), patch_size)  # Store patch as an object
            patches.append(patch)

    # Detect keypoints inside each patch
    for patch in patches:
        patch.detect_keypoints(fast)
        global_keypoints = patch.map_keypoints_to_global()

        # Store motion vectors (original keypoints to new keypoints)
        for gkp in global_keypoints:
            motion_arrows.append((patch.position, gkp))  # Start = patch center, End = detected feature

    # Arrange patches into a grid
    if patches:
        num_patches = len(patches)
        grid_size = int(np.ceil(np.sqrt(num_patches)))  # Calculate square grid size

        # Extract patch images for stacking
        patch_images = [patch.image for patch in patches]

        # Fill grid with blank patches if necessary
        while len(patch_images) < grid_size**2:
            patch_images.append(np.zeros((patch_size, patch_size, 3), dtype=np.uint8))

        # Stack patches into rows and then into a grid
        rows = [np.hstack(patch_images[i:i+grid_size]) for i in range(0, len(patch_images), grid_size)]
        stacked_patches = np.vstack(rows)  # Create final grid
    else:
        stacked_patches = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)  # Empty placeholder

    # Create a reconstructed frame (black canvas)
    reconstructed_frame = np.zeros_like(frame)

    # Place extracted patches at their original positions
    for patch in patches:
        x, y = patch.position
        reconstructed_frame[y:y+patch_size, x:x+patch_size] = patch.image  # Paste patch onto black canvas

    # Draw motion arrows (from original keypoints to detected keypoints inside patches)
    reconstructed_with_arrows = reconstructed_frame.copy()
    # for (start, end) in motion_arrows:
        # cv2.arrowedLine(reconstructed_with_arrows, start, end, (0, 255, 255), 2, tipLength=0.3)

     # --- Run ORB on the reconstructed frame ---
    gray_reconstructed = cv2.cvtColor(reconstructed_with_arrows, cv2.COLOR_BGR2GRAY)
    orb_keypoints = orb.detect(gray_reconstructed, None)  # ORB keypoint detection
    orb_keypoints = sorted(orb_keypoints, key=lambda kp: (kp.pt[1], kp.pt[0]))  # Sort ORB keypoints

    # Draw ORB keypoints on the reconstructed frame
    reconstructed_with_orb_keypoints = reconstructed_with_arrows.copy()
    cv2.drawKeypoints(reconstructed_with_arrows, orb_keypoints, reconstructed_with_orb_keypoints, color=(0, 0, 255), flags=0)

    # Display all three windows
    cv2.imshow("FAST Keypoints", frame_with_keypoints)
    cv2.imshow("Keypoint Patches Grid", stacked_patches)
    cv2.imshow("Reconstructed Image with Motion Arrows", reconstructed_with_arrows)
    cv2.imshow("Reconstructed with ORB Keypoints", reconstructed_with_orb_keypoints)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
