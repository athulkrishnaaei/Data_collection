import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

# Paths (Customize as per your folder structure)
image_folder = '/home/athul/Work/Data_collection/capture/detected_images'
label_folder = '/home/athul/Work/Data_collection/capture/detected_images/labels'

class ImageLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Apple Classifier")
        self.root.geometry("1200x900")  # Resize the window to fit larger images

        # Load the list of image files from the images folder
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
        self.current_index = 0
        self.current_image = None

        # Create GUI components
        self.image_label = tk.Label(root, text="Image will be here", width=100, height=100)  # Larger size
        self.image_label.grid(row=0, column=0, padx=20, pady=20, columnspan=3)  # Spanning multiple columns

        self.class_label = tk.Label(root, text="Class: ", font=("Helvetica", 14))
        self.class_label.grid(row=1, column=0, padx=20, pady=20, sticky="e")  # Align left

        self.class_entry = tk.Entry(root, font=("Helvetica", 14), width=10)
        self.class_entry.grid(row=1, column=1, padx=20, pady=20)

        self.filename_label = tk.Label(root, text="Filename: ", font=("Helvetica", 14))
        self.filename_label.grid(row=2, column=0, padx=20, pady=20, sticky="e")  # Align left

        self.instructions_label = tk.Label(root, text="Use 0 for normal, 1 for rotten apple.", font=("Helvetica", 12), fg="blue")
        self.instructions_label.grid(row=3, column=0, padx=20, pady=20, columnspan=3)  # Span across columns

        # Buttons in the center
        self.previous_button = tk.Button(root, text="Previous Image", command=self.previous_image, font=("Helvetica", 14), width=20)
        self.previous_button.grid(row=4, column=0, padx=20, pady=20)

        self.next_button = tk.Button(root, text="Next Image", command=self.next_image, font=("Helvetica", 14), width=20)
        self.next_button.grid(row=4, column=2, padx=20, pady=20)

        self.save_button = tk.Button(root, text="Save Label", command=self.save_label, font=("Helvetica", 14), width=20)
        self.save_button.grid(row=5, column=0, columnspan=3, pady=20)  # Span across all columns

        # Load the first image
        self.load_image()

    def load_image(self):
        # Get the current image and label filename
        image_file = self.image_files[self.current_index]
        label_file = os.path.splitext(image_file)[0] + '.txt'
        image_path = os.path.join(image_folder, image_file)

        # Load the image using PIL
        img = Image.open(image_path)
        img.thumbnail((1000, 700))  # Resize the image to fit better within the window
        img = ImageTk.PhotoImage(img)
        
        # Display the image
        self.image_label.config(image=img)
        self.image_label.image = img

        # Get the label (0 or 1) from the label file
        label_path = os.path.join(label_folder, label_file)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label_data = f.read().strip().split()
                label = label_data[0]  # The first element is the class (0 or 1)
        else:
            label = "0"  # Default label if not found (if file doesn't exist, it will be created later)

        # Update the class label and filename in the UI
        self.class_entry.delete(0, tk.END)
        self.class_entry.insert(0, label)
        self.filename_label.config(text=f"Filename: {image_file}")

    def save_label(self):
        # Get the new label from the entry box
        new_label = self.class_entry.get()
        if new_label not in ['0', '1']:
            print("Invalid label. Please enter 0 for normal or 1 for rotten.")
            return
        
        # Get the current image and label filename
        image_file = self.image_files[self.current_index]
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(label_folder, label_file)

        # Check if the label file exists, if not, create an empty label file
        if not os.path.exists(label_path):
            with open(label_path, 'w') as f:
                f.write(f"{new_label} 0.0 0.0 0.0 0.0 0.0\n")  # Create a new label with dummy bounding box data

        # Read the current label data (the entire line), then update the label
        with open(label_path, 'r') as f:
            label_data = f.read().strip().split()

        # Update the label to the new class (0 or 1)
        label_data[0] = new_label  # Update the class label (first element)

        # Save the updated label data
        with open(label_path, 'w') as f:
            f.write(' '.join(label_data))

        print(f"Label for {image_file} saved as {new_label}")

    def next_image(self):
        # Move to the next image
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.load_image()

    def previous_image(self):
        # Move to the previous image
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self.load_image()

if __name__ == "__main__":
    # Initialize Tkinter
    root = tk.Tk()
    app = ImageLabelingApp(root)
    root.mainloop()
