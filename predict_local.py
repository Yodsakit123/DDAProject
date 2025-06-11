import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms
from models import TransferNet

# Office-31 class names
class_names = [
    'back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator',
    'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', 'headphones',
    'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone', 'monitor',
    'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
    'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler',
    'tape_dispenser', 'trash_can'
]

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load multiple models
def load_model(path):
    m = TransferNet(num_class=31, base_net='resnet50', transfer_loss='lmmd', use_bottleneck=True)
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m

models = [
    load_model("dsan_amazon_to_webcam.pth"),
    load_model("dsan_webcam_to_amazon.pth"),
    load_model("dsan_dslr_to_webcam.pth")
]

# GUI setup
root = tk.Tk()
root.title("Image Recognition (Office-31 class)")
root.geometry("400x400")

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="")
result_label.pack()

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    display_img = img.copy()
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        all_outputs = []
        for model in models:
            output = model.predict(img_tensor)
            all_outputs.append(torch.nn.functional.softmax(output, dim=1))
        avg_output = torch.stack(all_outputs).mean(dim=0)
        _, pred = torch.max(avg_output, 1)
        score = avg_output[0][pred.item()]
    
    return class_names[pred.item()], score.item(), display_img

def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        pred_class, score, img = predict_image(file_path)
        result_label.config(text=f"Predicted: {pred_class} ({score*100:.2f}%)")
        tk_img = ImageTk.PhotoImage(img)
        image_label.config(image=tk_img)
        image_label.image = tk_img

btn = tk.Button(root, text="Choose Image", command=browse_file)
btn.pack()

root.mainloop()
