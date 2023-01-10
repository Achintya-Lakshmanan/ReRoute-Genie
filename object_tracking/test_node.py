import cv2
import numpy as np
import random

# Define a function to calculate the speed of a moving vehicle
def estimate_speed(centroid, frames_elapsed):
    # Define a conversion factor for pixels per frame to miles per hour
    mph_per_pixel_per_frame = 0.1
    # Calculate the speed in pixels per frame
    pixel_speed = np.sqrt(np.sum(np.diff(centroid) ** 2)) / frames_elapsed
    # Convert the speed from pixels per frame to miles per hour
    speed = pixel_speed * mph_per_pixel_per_frame
    return speed

# Define the video file to process
video_file = "/Users/daver/Desktop/temp_3/Object tracking/No congestion.mp4"
# Open the video file
cap = cv2.VideoCapture(video_file)
 
# Get the first frame
ret, first_frame = cap.read()
# Convert the first frame to grayscale
first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Set up an empty list to store the centroids of the vehicles
centroids = []
# Set the frames elapsed to 0
frames_elapsed = 0

# Loop through the frames in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames_elapsed += 1
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate the absolute difference between the current frame and the first frame
    diff = cv2.absdiff(first_frame_gray, gray)
    # Threshold the difference image to create a binary image
    _, thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    # Dilate the binary image to fill in small holes
    dilated = cv2.dilate(thresholded, None, iterations=2)
    # Find the contours in the binary image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Iterate through the contours
    for contour in contours:
        # Calculate the moment of the contour
        moment = cv2.moments(contour)
        if moment["m00"] > 1000:
            # Calculate the x,y coordinate of the centroid
            x = int(moment["m10"] / moment["m00"])
            y = int(moment["m01"] / moment["m00"])
            # Append the centroid to the list of centroids
            centroids.append((x, y))
            # Draw a circle at the centroid
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    # Show the current frame
    cv2.imshow("Frame", frame)
    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the video file
cap.release()

# Calculate the speed of the vehicle using the centroids
if len(centroids) > 2:
    # Pick three random centroids
    random_centroids = random.sample(centroids, 3)
    speeds = []
    for centroid in random_centroids:
        speed = estimate_speed(centroid, frames_elapsed)
        speeds.append(speed)
    # Print the average speed
    print("Average speed: {:.2f} mph".format(sum(speeds) / len(speeds)))
else:
    print("Not enough centroids detected.")

# Close all windows
cv2.destroyAllWindows()


def congestion_video() :
    # Define the video file to process
    video_file = "/Users/daver/Desktop/temp_3/Object tracking/No congestion.mp4"
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Get the first frame
    ret, first_frame = cap.read()
    # Convert the first frame to grayscale
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Set up an empty list to store the centroids of the vehicles
    centroids = []
    # Set the frames elapsed to 0
    frames_elapsed = 0

    # Loop through the frames in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames_elapsed += 1
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate the absolute difference between the current frame and the first frame
        diff = cv2.absdiff(first_frame_gray, gray)
        # Threshold the difference image to create a binary image
        _, thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        # Dilate the binary image to fill in small holes
        dilated = cv2.dilate(thresholded, None, iterations=2)
        # Find the contours in the binary image
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Iterate through the contours
        for contour in contours:
            # Calculate the moment of the contour
            moment = cv2.moments(contour)
            if moment["m00"] > 1000:
                # Calculate the x,y coordinate of the centroid
                x = int(moment["m10"] / moment["m00"])
                y = int(moment["m01"] / moment["m00"])
                # Append the centroid to the list of centroids
                centroids.append((x, y))
                # Draw a circle at the centroid
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        # Show the current frame
        cv2.imshow("Frame", frame)
        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the video file
    cap.release()

    # Calculate the speed of the vehicle using the centroids
    if len(centroids) > 2:
        # Pick three random centroids
        random_centroids = random.sample(centroids, 3)
        speeds = []
        for centroid in random_centroids:
            speed = estimate_speed(centroid, frames_elapsed)
            speeds.append(speed)
        # Print the average speed
        print("Average speed: {:.2f} mph".format(sum(speeds) / len(speeds)))
    else:
        print("Not enough centroids detected.")

    # Close all windows
    cv2.destroyAllWindows()
    return 1
