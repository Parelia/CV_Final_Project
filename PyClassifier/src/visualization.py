###########################
# Author: Capuzzo Daniele #
###########################

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import (
    classification_report as sk_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

def plot_class_distribution(train_class_counts, val_class_counts):
    # bar diagram of training and validation classes distribution
    labels_map = {0: 'Parasitized', 1: 'Uninfected'}

    # Prepare data
    x_labels = [labels_map[i] for i in sorted(labels_map.keys())]
    x_pos = range(len(x_labels))
    train_vals = [train_class_counts[i] for i in sorted(labels_map.keys())]
    val_vals = [val_class_counts[i] for i in sorted(labels_map.keys())]

    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))

    # Plot bars
    bar_width = 0.35
    bars1 = plt.bar([i - bar_width/2 for i in x_pos], train_vals, width=bar_width, label='Training', color='skyblue')
    bars2 = plt.bar([i + bar_width/2 for i in x_pos], val_vals, width=bar_width, label='Test', color='orange')

    # Add counts on top of bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 100, f'{height}', ha='center', va='bottom', fontsize=9)

    # final touches
    plt.xticks(ticks=x_pos, labels=x_labels)
    plt.ylabel("Number of Images")
    plt.title("Class Distribution in Training vs Test Set")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_learning_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def classification_report(y_true, y_pred):
    print(sk_report(y_true, y_pred, labels=[0, 1], target_names=['Uninfected', 'Parasitized']))

def plot_roc(y_true, y_probs):
    print(f"AUC-ROC: {roc_auc_score(y_true, y_probs):.4f}")
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_probs):.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Uninfected', 'Parasitized'],
                yticklabels=['Uninfected', 'Parasitized'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()