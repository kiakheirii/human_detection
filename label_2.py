import os
import cv2

# Directory containing images
image_dir = 'D:\\30fps_frames'

# Directory to store labeled images
output_dir = 'labeled_images'

# Create directories for each class
class_dirs = {
    0: os.path.join(output_dir, '0'),
    1: os.path.join(output_dir, '1'),
}

for class_dir in class_dirs.values():
    os.makedirs(class_dir, exist_ok=True)

# Track already processed images
processed_files = set()
for class_dir, path in class_dirs.items():
    processed_files.update(os.listdir(path))

# Loop over images
for img_file in sorted(os.listdir(image_dir)):
    # Skip already processed files
    if img_file in processed_files:
        continue

    img_path = os.path.join(image_dir, img_file)
    img = cv2.imread(img_path)

    # Skip if image is not readable
    if img is None:
        print(f"Cannot read image: {img_file}")
        continue

    # Set up display window
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    h, w, _ = img.shape
    cv2.resizeWindow("Image", w, h)

    # Display the image
    cv2.imshow("Image", img)
    print(f"Label the image: {img_file}")
    print("Press '0' for class 0, '1' for class 1, or 'q' to quit.")

    # Wait for user input
    key = cv2.waitKey(0)
    if key == ord('0' or '9'):
        label = 0
    elif key == ord('1' or '2'):
        label = 1
    elif key == ord('q'):
        print("Quitting and saving progress...")
        break
    else:
        print("Invalid input. Skipping image.")
        continue

    # Save the image in the respective labeled folder
    output_path = os.path.join(class_dirs[label], img_file)
    cv2.imwrite(output_path, img)
    print(f"Saved {img_file} to class {label} folder.")

# Cleanup
cv2.destroyAllWindows()
print("Labeling complete. Images saved to:", output_dir)
