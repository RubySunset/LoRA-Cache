import os
from tkinter import Tk, Button, Label, Frame
from PIL import Image, ImageTk

class ImageSlideshow:
    def __init__(self, master, image_groups):
        self.master = master
        self.image_groups = image_groups
        self.current_group = 0

        self.master.title("Image Slideshow")

        self.frame = Frame(master)
        self.frame.pack()

        self.image_labels = [Label(self.frame) for _ in range(5)]
        self.filename_labels = [Label(self.frame) for _ in range(5)]
        for i in range(5):
            self.image_labels[i].grid(row=(i // 3) * 2, column=i % 3, padx=5, pady=5)
            self.filename_labels[i].grid(row=(i // 3) * 2 + 1, column=i % 3, padx=5, pady=5)

        self.button_frame = Frame(master)
        self.button_frame.pack(pady=10)

        self.prev_button = Button(self.button_frame, text="<< Previous", command=self.show_prev_group)
        self.prev_button.pack(side="left", padx=10)

        self.next_button = Button(self.button_frame, text="Next >>", command=self.show_next_group)
        self.next_button.pack(side="left", padx=10)

        self.show_group(self.current_group)

    def show_group(self, group_index):
        group = self.image_groups[group_index]
        for i, image_path in enumerate(group):
            image = Image.open(image_path)
            image.thumbnail((300, 300))  # Resize to fit the grid
            photo = ImageTk.PhotoImage(image)
            self.image_labels[i].configure(image=photo)
            self.image_labels[i].image = photo  # Keep a reference to avoid garbage collection
            self.filename_labels[i].configure(text=os.path.basename(image_path))
        for i in range(len(group), 5):
            self.image_labels[i].configure(image='')
            self.filename_labels[i].configure(text='')

    def show_next_group(self):
        self.current_group = (self.current_group + 1) % len(self.image_groups)
        self.show_group(self.current_group)

    def show_prev_group(self):
        self.current_group = (self.current_group - 1) % len(self.image_groups)
        self.show_group(self.current_group)

def load_image_groups(directory, group_size=5):
    image_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.png')]
    image_files.sort()
    image_groups = [image_files[i:i + group_size] for i in range(0, len(image_files), group_size)]
    return image_groups

if __name__ == "__main__":
    root = Tk()
    directory = "images"  # Change this to your directory
    image_groups = load_image_groups(directory)
    if not image_groups:
        print("No images found in the directory.")
    else:
        slideshow = ImageSlideshow(root, image_groups)
        root.mainloop()