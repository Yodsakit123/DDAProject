import pandas as pd
import matplotlib.pyplot as plt

# Load your training log
log_df = pd.read_csv('A-Wbnm_log.csv')  # Use your actual file name

# Plot Classification Loss, Transfer Loss, Total Loss
plt.figure(figsize=(10, 6))
plt.plot(log_df['epoch'], log_df['cls_loss'], label='Classification Loss')
plt.plot(log_df['epoch'], log_df['transfer_loss'], label='Transfer Loss')
plt.plot(log_df['epoch'], log_df['total_loss'], label='Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Test Accuracy
plt.figure(figsize=(8, 5))
plt.plot(log_df['epoch'], log_df['test_acc'], color='green', marker='o', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Over Epochs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
