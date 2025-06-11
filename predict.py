import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models import TransferNet
import os
import sys

# List of Office-31 class names
class_names = [
    'back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator',
    'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', 'headphones',
    'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone', 'monitor',
    'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
    'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler',
    'tape_dispenser', 'trash_can'
]

# Image transformation
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

# Prediction function
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
    
    return class_names[pred.item()], score.item(), img

# Image usage

imagename = input("Please enter image file name (include .jpg): ")
image_path = "/content/24-25_CE301_kittichalermroj_yodsakit/" + imagename

if os.path.exists(image_path):
    print("Image found:", image_path)
else:
    print("Wrong image file name or file not found.")
    sys.exit()
predicted_class, confidence, image = predict_image(image_path)
print(f"Predicted Class: {predicted_class} ({confidence * 100:.2f}%)")

# Save the prediction as an image
plt.figure()
plt.imshow(image)
plt.title(f"Prediction: {predicted_class} ({confidence * 100:.2f}%)")
plt.axis('off')
plt.savefig("prediction_result.png")  # Saves the image to file
print("Prediction result saved as 'prediction_result.png'")

plt.show()