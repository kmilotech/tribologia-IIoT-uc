import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc,f1_score,recall_score,accuracy_score,precision_score,roc_auc_score

'''
Funcion para identificar elementos tipo objeto que no son
convertibles a float
'''
def found_non_float(df, columna):
    no_convertibles = []
    for index, valor in enumerate(df[columna]):
        try:
            float(valor)
        except ValueError:
            no_convertibles.append((index, valor))
    
    return no_convertibles




def plot_confusion_matrix(y_true, y_pred, labels=None, cmap='Blues', normalize=False):
    """
    Plots the confusion matrix for given true and predicted labels.

    Parameters:
    y_true (list or array-like): True labels.
    y_pred (list or array-like): Predicted labels.
    labels (list, optional): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
    cmap (str, optional): Colormap used for the heatmap. Default is 'Blues'.
    normalize (bool, optional): Whether to normalize the confusion matrix. Default is False.

    Returns:
    None
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
    else:
        title = 'Confusion Matrix'

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_confusion_matrix_with_percentage(y_true, y_pred, labels=None, cmap='Blues'):
    """
    Plots the confusion matrix with percentage distribution for given true and predicted labels.

    Parameters:
    y_true (list or array-like): True labels.
    y_pred (list or array-like): Predicted labels.
    labels (list, optional): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
    cmap (str, optional): Colormap used for the heatmap. Default is 'Blues'.

    Returns:
    None
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_percentage = cm.astype('float') / cm.sum() * 100

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix with Percentage Distribution')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_classification_report(y_true, y_pred, labels=None, cmap='Blues'):
    """
    Plots the classification report for given true and predicted labels.

    Parameters:
    y_true (list or array-like): True labels.
    y_pred (list or array-like): Predicted labels.
    labels (list, optional): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
    cmap (str, optional): Colormap used for the heatmap. Default is 'Blues'.

    Returns:
    None
    """
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    plt.figure(figsize=(10, 7))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap=cmap, cbar=False)
    plt.title('Classification Report')
    plt.ylabel('Metrics')
    plt.xlabel('Classes')
    plt.show()


def plot_roc_curve(y_true, y_proba, pos_label=1):
    """
    Plots the ROC curve for given true labels and predicted probabilities.

    Parameters:
    y_true (list or array-like): True binary labels.
    y_proba (list or array-like): Predicted probabilities or decision function.
    pos_label (int, optional): The label of the positive class. Default is 1.

    Returns:
    None
    """
    # Compute the ROC curve and AUC score
    fpr, tpr, thresholds = roc_curve(y_true, y_proba, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def model_metrics(model_name,y_test,y_pred):
    dict_to_list={
                    'F1-Score':round(f1_score(y_test, y_pred, average='micro'),2),
                    'Recall': round(recall_score(y_pred, y_test),2),
                    'Precision': round(precision_score(y_pred, y_test),2),
                    'Accuracy': round(accuracy_score(y_pred, y_test),2),
                     'auc' : roc_auc_score(y_test, y_pred)
                }
    return [model_name, {'metrics':dict_to_list}]

def  model_persist(lst,model_name,y_test,y_pred):
        for i in range(len(lst)):
            print(len(lst))
            if model_name ==lst[i][0]:
                lst.remove(lst[i])
                break
        lst.append(model_metrics(model_name,y_test,y_pred))

