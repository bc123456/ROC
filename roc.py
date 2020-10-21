from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback

def print_metric_results(accuracy, auc, cm):
    
    print(f'Accuracy: {accuracy:.3f}' )
    print(f'AUC: {auc:.3f}')
    print(f'Confusion Matrix:')
    print(cm)
    
def calculate_best_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=True)
    # method to determine the best threshold
    # now using sum of sensitivity and specificity
    score = tpr + (1 - fpr)
    best_threshold_id = np.argmax(score)
    best_threshold = thresholds[best_threshold_id]
    best_fpr = fpr[best_threshold_id]
    best_tpr = tpr[best_threshold_id]
    
    return best_fpr, best_tpr, best_threshold

def construct_plotly_graph(fpr1, tpr1, thresholds1, best_fpr1, best_tpr1, auc1,
                           fpr2, tpr2, thresholds2, best_fpr2, best_tpr2, auc2):
    
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Train ROC curve, AUC = {auc1:.4f}', f'Val/Test ROC curve, AUC = {auc2:.4f}'),
    )
    
    # the first plot (train dataset)
    fig.add_trace(
        go.Scatter(
            name='',
            x=fpr1, y=tpr1,
            fill='tozeroy',
            text=thresholds1,
            hovertemplate='Threshold = %{text:.6f}',
        ),
        row=1, col=1,
    )
    
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1,
        row=1, col=1
    )
    
    fig.add_annotation(
        x=best_fpr1,
        y=best_tpr1,
        text=f"Best threshold <br>({best_fpr1:.4f}, {best_tpr1:.4f})",
        row=1, col=1
    )
    
    # the second plot (test dataset)
    fig.add_trace(
        go.Scatter(
            name='',
            x=fpr2, y=tpr2,
            fill='tozeroy',
            text=thresholds1,
            hovertemplate='Threshold = %{text:.6f}',
        ),
        row=1, col=2,
    )
    
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1,
        row=1, col=2
    )
    
    fig.add_annotation(
        x=best_fpr2,
        y=best_tpr2,
        text=f"Best threshold <br>({best_fpr2:.4f}, {best_tpr2:.4f})",
        row=1, col=2
    )
    
    # Final plot adjustments
    fig.update_layout(
        width=1000,
        height=500,
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])    
    
    fig.show()
    
    return fig

def construct_graph(fpr, tpr, best_fpr, best_tpr, auc, ax=None):
    ax = ax or plt.gca()
    ax.plot(fpr, tpr),
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.plot(best_fpr, best_tpr, marker='o', color='black')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC curve, AUC = {auc:.4f}')
    return ax

def plot_ROC(y_train_true, y_train_prob, y_test_true, y_test_prob):
    '''
    a funciton to plot the ROC curve for train labels and test labels.
    Use the best threshold found in train set to classify items in test set.
    '''
    
    # Train stats
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train_true, y_train_prob, pos_label =True)
    best_fpr_train, best_tpr_train, best_threshold = calculate_best_threshold(y_train_true, y_train_prob)
    auc_train = roc_auc_score(y_train_true, y_train_prob)
    y_train_pred = y_train_prob > best_threshold
    
    print('Train results')
    print_metric_results(
        accuracy_score(y_train_true, y_train_pred),
        auc_train,
        confusion_matrix(y_train_true, y_train_pred)         
    )
    
    # Test stats
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test_true, y_test_prob, pos_label =True)
    auc_test = roc_auc_score(y_test_true, y_test_prob)
    y_test_pred = y_test_prob > best_threshold
    cm_test = confusion_matrix(y_test_true, y_test_pred)
    
    print('Valid/Test results')
    print_metric_results(
        accuracy_score(y_test_true, y_test_pred),
        auc_test,
        cm_test        
    )
    
    best_tpr_test = recall_score(y_test_true, y_test_pred)
    best_fpr_test = float(cm_test[0][1])/(cm_test[0][0]+ cm_test[0][1])
    
    # try to plot using plotly first, if not successful, use matplolib
    try:
        fig = construct_plotly_graph(
            fpr_train, tpr_train, thresholds_train, best_fpr_train, best_tpr_train, auc_train,
            fpr_test, tpr_test, thresholds_test, best_fpr_test, best_tpr_test, auc_test
        )
    except Exception as e:
        traceback.print_exc()
    
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1 = construct_graph(fpr_train, tpr_train, best_fpr_train, best_tpr_train, auc_train, ax=ax1)
        ax2 = construct_graph(fpr_test, tpr_test, best_fpr_test, best_tpr_test, auc_test, ax=ax2)
        plt.show()
    
    print(f'Best Threshold: {best_threshold}')
    
    return best_threshold