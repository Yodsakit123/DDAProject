import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the CSV
df = pd.read_csv('bnm_amazon_dslr_confusion.csv')

# Get true and predicted labels
true_labels = df['true_label']
pred_labels = df['predicted_label']

# Define Office-31 class names (in order)
class_names = [
    'back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator',
    'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', 'headphones',
    'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone', 'monitor',
    'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
    'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler',
    'tape_dispenser', 'trash_can'
]

# Generate confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot the heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (BNM - Amazon to DSLR)')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
