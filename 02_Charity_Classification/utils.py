"""docstring."""

# Core
import string
from itertools import cycle

# Visalization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
# Confusion Matrix
# Precision vs Recall Curve
# Metrics
from sklearn.metrics import ConfusionMatrixDisplay  # Metrics
from sklearn.metrics import (PrecisionRecallDisplay, accuracy_score, auc,
                             average_precision_score, classification_report,
                             f1_score, make_scorer, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
# Optimization
from sklearn.model_selection import \
    GridSearchCV  # Find optimized hyperparameters
from sklearn.preprocessing import label_binarize


def centered(content: str) -> None:
    """Centralize text print.

    :param content: Text to be centralized
    """
    display(HTML(f"<div style='text-align:center'>{content}</div>"))


def data_check(df: pd) -> None:
    """Display general data information.

    :param df: _description_
    """
    print('='*149)
    centered('Data Information')
    display(df.info())

    print('='*149)
    centered('Check for nan values and duplicated rows')
    print(f'Number of nan values: {df.isna().sum().sum()}')
    print(f'Number of duplicated rows: {df.duplicated().sum()}')

    print('='*149)
    centered('Data Description')
    display(df.describe().T)

    print('='*149)
    centered('Data Head')
    display(df.head())

    print('='*149)
    centered('Data Shape')
    print(f'Data Shape: {df.shape}')


def remove_punctuation(txt: str) -> str:
    """Remove ponctuation on the description text.

    :param txt: Text with ponctuation
    :return: Text without ponctuation
    """
    return txt.translate(str.maketrans('', '', string.punctuation)).lower()


def model_metrics(label: pd, pred: pd) -> pd:
    """Show the result metrics of a model.

    :param label: Ground Truth
    :param pred: Predicions
    :return: Result metrics
    """
    # Compute metrics
    acc = accuracy_score(label, pred)
    recall = recall_score(label, pred, average='macro')
    precision = precision_score(label, pred, average='macro')
    f1 = f1_score(label, pred, average='macro')

    # Metrics to dictionary
    metrics_dict = {'Accuracy': acc,
                    'Recall': recall,
                    'Precision': precision,
                    'F1-score': f1}

    # creating a dataframe of metrics
    metrics_df = pd.DataFrame(metrics_dict, index=[0])

    return metrics_df


def pr_curve(model_name: str, target: pd, pred_prob: list) -> None:
    """Plot the Precision-Recall curve.

    :param model_name: Model name
    """
    classes = np.unique(target)
    n_classes = classes.shape[0]
    lw = 1

    # roc curve for classes
    fpr = {}  # Fpr = Fp/Fp+Tn
    tpr = {}  # recall (Tp/Tp+Fn)
    thresh = {}
    roc_auc = dict()

    n_class = classes.shape[0]

    y_test_binarized = label_binarize(target, classes=np.unique(target))
    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(
            y_test_binarized[:, i], pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label='LG_ROC curve of class {0} (area = {1:0.2f})'.format(
                i, roc_auc[i])
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(
        f'Receiver Operating Characteristic for {model_name} Classification ')
    plt.legend(loc='lower right')
    plt.show()

    print('AUC: ' + str(roc_auc[i]))
    # run.log('AUC', np.float(roc_auc[i]))


def results(
    model: object,
    params: dict,
    train: pd,
    test: pd,
    features: list,
    target: str,
    metric: object,
    metric_name: 'str',
    search_params: bool,
) -> object:
    """Apply ML models and show the results.

    :param model: Machine learning model
    :param params: Parameters of the model
    :param train: Training set
    :param test: Testing set
    :param features: Features used as input
    :param target: Target used as output
    :param metric: Metric object applied for results
    :param metric_name: Metric name applied for results
    :param search_params: Boolean to set the usage of search parameters
    :return: final model object
    """
    if search_params:
        # Grid Search
        grid_cv = GridSearchCV(model, params, scoring=metric,
                               cv=5, error_score='raise').fit(train[features], train[target])

        # Best Model
        best_params = grid_cv.best_params_
        print(
            f'Best parameters are {best_params}\n{metric_name} score: {grid_cv.best_score_}!\n\n')
    else:
        best_params = params

    # Training model
    model = model.set_params(**best_params)
    model.fit(train[features], train[target])

    # Computing predictions
    pred_train = model.predict(train[features])
    pred_test = model.predict(test[features])
    # pred_proba = model.predict_proba(test[features])

    # Metrics
    print(50 * '*')
    print('Training metrics:')
    print(model_metrics(train[target], pred_train))
    print(50 * '*')
    print('Test metrics:')
    print(model_metrics(test[target], pred_test))
    print(50 * '*')
    print(5*'\n')
    print(50 * '*')
    print('Training metrics per class:')
    print(classification_report(train[target], pred_train))
    print(50 * '*')
    print('Test metrics per class:')
    print(classification_report(test[target], pred_test))
    print(50 * '*')

    # Confusion matrix:
    plt.rcParams.update({'font.size': 10, 'figure.figsize': (10, 10)})
    labels = [
        'Human Services',
        'Health',
        'Education',
        'Arts, Culture, Humanities',
        'Religion',
        'Research and Public Policy',
        # 'International',
        'Community Development',
        'Animals',
        'Human and Civil Rights',
        'Environment'
    ]
    # Training Data
    ConfusionMatrixDisplay.from_predictions(
        train[target], pred_train,
        cmap=plt.cm.Blues,
        display_labels=labels)
    plt.title('Training Data')
    plt.xticks(rotation=45, ha='right')
    # plt.yticks(rotation=45, ha='right');
    plt.grid(False)
    plt.tight_layout()

    # Test Data
    ConfusionMatrixDisplay.from_predictions(
        test[target], pred_test,
        cmap=plt.cm.Blues,
        display_labels=labels)
    plt.title('Testing Data')
    plt.xticks(rotation=45, ha='right')
    # plt.yticks(rotation=45, ha='right');
    plt.grid(False)
    plt.tight_layout()
    return pred_test
