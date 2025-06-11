from sklearn.metrics import classification_report
import pandas as pd

# Load your prediction results
df = pd.read_csv('bnm_amazon_dslr_confusion.csv')

true_labels = df['true_label']
pred_labels = df['predicted_label']

# Optional: class names (if you want human-readable labels)
class_names = [
    'back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator',
    'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', 'headphones',
    'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone', 'monitor',
    'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
    'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler',
    'tape_dispenser', 'trash_can'
]

# Generate the classification report
report = classification_report(true_labels, pred_labels, target_names=class_names, digits=4)

print(report)
