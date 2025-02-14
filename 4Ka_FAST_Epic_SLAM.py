import cv2
import numpy as np
import time  # Import time module
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Change if using an external webcam

# Create a FAST detector
fast = cv2.FastFeatureDetector_create(threshold=35, nonmaxSuppression=True)
orb = cv2.ORB_create(nfeatures=1000)

# Define a margin (tolerance) in pixels
initial_margin = 5  # Change this to your desired margin (tolerance)
max_margin = 50

# List to store matching percentages at different margins for each frame
all_matching_percentages = []

def euclidean_distance(kp1, kp2):
    """Calculate the Euclidean distance between two keypoints (tuples)."""
    return math.sqrt((kp1[0] - kp2[0])**2 + (kp1[1] - kp2[1])**2)

def graph_percentage_pairs():
    # Read the CSV file into a DataFrame
    df = pd.read_csv("matching_percentages.csv")
    
    # Get unique margins, sorted
    margin_values = sorted(df["Margin"].unique())  # Unique margin values sorted
    
    # Calculate the number of rows and columns for the subplot grid
    num_plots = len(margin_values)
    num_cols = 3  # We'll set the number of columns to 3 (you can adjust this)
    num_rows = math.ceil(num_plots / num_cols)  # Calculate the necessary number of rows
    
    # Create the figure and subplots grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    
    # Flatten the axes array for easier indexing, in case we have a 2D grid
    axes = axes.flatten()

    # Loop over each margin value and plot
    for i, margin in enumerate(margin_values):
        # Filter the data for this margin
        margin_data = df[df["Margin"] == margin]["Matching Percentage"]
        
        # Plot histogram for this margin
        axes[i].hist(margin_data, bins=10, edgecolor='black', color='skyblue')
        axes[i].set_xlabel("Matching Percentage (%)")
        axes[i].set_ylabel("Frequency")
        axes[i].set_title(f"Margin: {margin} Pixels")

        # Set the x-axis limit to 0-100 (fixed scale)
        axes[i].set_xlim(0, 100)

        # Add KDE (interpolated line) for the matching percentage data
        sns.kdeplot(margin_data, ax=axes[i], color='red', lw=2)  # red line for the KDE
    
    # Hide any empty subplots (if there are any)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')  # Turn off unused subplots

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust both vertical and horizontal spacing

    # Save the figure to a file
    figure_filename = 'margin_matching_percentage_distribution.png'
    plt.savefig(figure_filename, bbox_inches='tight')  # Save the figure with tight layout to avoid clipping
    print(f"Figure saved as {figure_filename}")

    # Show the plot
    plt.show()


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


# --- MAIN ---
def main():

    # Set a limit on the number of frames (iterations)
    frame_count = 0
    max_frames = 10

    print(f"Number of max frames (iterations): {max_frames}")

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

        og_orb_keypoints = orb.detect(gray, None)  # Original ORB Keypoint detection
        # print(f"FAST Keypoint Extraction Time: {elapsed_time:.6f} seconds")  # Print result
        # -------------------------------------------------

        # Sort keypoints in a stable way (top-left to bottom-right)
        keypoints = sorted(keypoints, key=lambda kp: (kp.pt[1], kp.pt[0]))

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

        # Create a reconstructed frame (black canvas)
        reconstructed_frame = np.zeros_like(frame)

        # Place extracted patches at their original positions
        for patch in patches:
            x, y = patch.position
            reconstructed_frame[y:y+patch_size, x:x+patch_size] = patch.image  # Paste patch onto black canvas

        # --- Run ORB on the reconstructed frame ---
        gray_reconstructed = cv2.cvtColor(reconstructed_frame, cv2.COLOR_BGR2GRAY)
        recon_orb_keypoints = orb.detect(gray_reconstructed, None)  # ORB keypoint detection
        recon_orb_keypoints = sorted(recon_orb_keypoints, key=lambda kp: (kp.pt[1], kp.pt[0]))  # Sort ORB keypoints
        
        # Draw ORB keypoints on the reconstructed frame
        reconstructed_with_orb_keypoints = reconstructed_frame.copy()
        cv2.drawKeypoints(reconstructed_frame, recon_orb_keypoints, reconstructed_with_orb_keypoints, color=(0, 0, 255), flags=0)

        # --- Store Original and Reconstructed ORB Keypoints in Lists ---
        og_orb_keypoints_list = [kp.pt for kp in og_orb_keypoints]  # List of original ORB keypoints
        recon_orb_keypoints_list = [kp.pt for kp in recon_orb_keypoints]  # List of reconstructed ORB keypoints

        # --- Calculate Matching Percentages ---
        matching_data = []  # List to store margin and matching percentage pairs for every frame
        
        for margin in range(initial_margin, max_margin + 1, 5):
            matching_keypoints_within_margin = []
            
            for og_kp in og_orb_keypoints:
                og_pt = og_kp.pt  # (x, y) coordinates of original keypoint
            
                for recon_kp in recon_orb_keypoints:
                    recon_pt = recon_kp.pt  # (x, y) coordinates of reconstructed keypoint
        
                    # Calculate Euclidean distance between keypoints
                    distance = euclidean_distance(og_pt, recon_pt)
                    
                    # Check if the distance is within the margin
                    if distance <= margin:
                        matching_keypoints_within_margin.append(og_kp)
                        break  # Once a match is found, no need to compare further
        
            # Calculate matching percentage
            matching_keypoints = len(matching_keypoints_within_margin)
            total_keypoints = len(og_orb_keypoints)
            
            matching_percentage = (matching_keypoints / total_keypoints) * 100 if total_keypoints > 0 else 0
            matching_data.append([margin, matching_percentage])  # Store both margin and matching percentage
        
        # Append the current frame's matching data to the overall list
        all_matching_percentages.append(matching_data)
        
        # Increment frame count and check if we've reached the maximum frames
        frame_count += 1
        if frame_count >= max_frames:
            print("Reached the maximum number of frames. Exiting the loop.")
            break

    # --- Save data to CSV ---
    flattened_data = []
    for frame_data in all_matching_percentages:
        for margin, matching_percentage in frame_data:
            flattened_data.append([margin, matching_percentage])

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(flattened_data, columns=["Margin", "Matching Percentage"])
    df.to_csv("matching_percentages.csv", index=False)
    print("Data saved to matching_percentages.csv")

    # --- Plot the histogram ---
    graph_percentage_pairs()

    # Display all three windows
    # cv2.imshow("FAST Keypoints", frame_with_keypoints)
    # cv2.imshow("Keypoint Patches Grid", stacked_patches)
    # cv2.imshow("Reconstructed Image with Motion Arrows", reconstructed_with_arrows)
    # cv2.imshow("Reconstructed with ORB Keypoints", reconstructed_with_orb_keypoints)

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()




