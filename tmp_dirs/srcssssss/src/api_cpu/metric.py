from sklearn.metrics import f1_score, roc_auc_score

def get_f1(y_true, y_pred, threshold=0.5):
    return f1_score(y_true, y_pred > threshold)

def get_auc(y_true, y_pred, threshold=0.5):
    return roc_auc_score(y_true, y_pred)