###########################
# Author: Capuzzo Daniele #
###########################

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

import config

from sklearn.metrics import (
    classification_report as sk_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

def save_plot(filename):
    if not os.path.exists(config.image_directory):
        os.makedirs(config.image_directory, exist_ok=True)
    
    path = os.path.join(config.image_directory, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {path}")

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
    bars2 = plt.bar([i + bar_width/2 for i in x_pos], val_vals, width=bar_width, label='Validation', color='orange')

    # Add counts on top of bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 100, f'{height}', ha='center', va='bottom', fontsize=9)

    # final touches
    plt.xticks(ticks=x_pos, labels=x_labels)
    plt.ylabel("Number of Images")
    plt.title("Class Distribution in Training vs Validation Set")
    plt.legend()
    plt.tight_layout()
    save_plot("class_distribution.png")
    plt.show()

def plot_learning_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1) # should start from 1 (not 0)

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

    save_plot("learning_curves.png")
    plt.show()

def classification_report(y_true, y_pred):
    print(sk_report(y_true, y_pred, labels=[0, 1], target_names=['Parasitized', 'Uninfected']))

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

    save_plot("roc.png")
    plt.show()

def plot_cm(y_true, y_pred):
    
    # beware of class ordering: 0 is Parasitized, 1 is Uninfected
    target_names = ['Parasitized', 'Uninfected'] 
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    save_plot("cm.png")
    plt.show()

def plot_prediction_samples(test_set, y_true, y_pred, y_probs):
    
    # flatten
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    y_probs = np.array(y_probs).ravel()
    
    # get images
    all_images = []
    for x, _ in test_set:
        all_images.append(x.numpy())
    images = np.concatenate(all_images, axis=0)
    
    # fix rendering
    images = np.clip(images, 0, 255).astype(np.uint8)
    
    # 0 parasitized, 1 uninfected
    idx_corr_par = np.where((y_true == 0) & (y_pred == 0))[0]
    idx_corr_unf = np.where((y_true == 1) & (y_pred == 1))[0]
    idx_wrong    = np.where(y_true != y_pred)[0]

    # take these positions
    positions = [0, 3, 6, 9]
    
    # get them
    samp_par = [idx_corr_par[p] for p in positions if p < len(idx_corr_par)]
    samp_unf = [idx_corr_unf[p] for p in positions if p < len(idx_corr_unf)]
    samp_wrong = [idx_wrong[p] for p in positions if p < len(idx_wrong)]

    samples = [
        (samp_par, "Correct Parasitized (True: 0, Pred: 0)"),
        (samp_unf, "Correct Uninfected (True: 1, Pred: 1)"),
        (samp_wrong, "Incorrect Predictions")
    ]

    fig = plt.figure(figsize=(14, 10))
    
    for row_idx, (indices, category) in enumerate(samples):
        for col_idx, i in enumerate(indices):
            ax = plt.subplot(3, 4, row_idx * 4 + col_idx + 1)
            plt.imshow(images[i]) 
            
            title = f"Actual: {int(y_true[i])} | Pred: {int(y_pred[i])}\nOutput: {y_probs[i]:.4f}"
            plt.title(title, fontsize=12)
            plt.axis('off')
            
    plt.subplots_adjust(hspace=0.5, wspace=0.1)
    
    # save image
    save_plot("gt_predictions.png")
    plt.show()