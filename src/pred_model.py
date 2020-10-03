from sklearn.base import clone 
from sklearn import metrics
import pandas as pd 
import seaborn as sns
from sklearn import tree

# function for creating a feature importance dataframe
def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
    return df

# plotting a feature importance dataframe (horizontal barchart)
def var_imp_plot(imp_df, title, y_sz = 8):
    imp_df.columns = ['feature', 'feature_importance']
    sns.barplot(x = 'feature_importance', y = 'feature', data = imp_df, orient = 'h', color = 'royalblue') \
       .set_title(title, fontsize = 20)
    plt.rcParams["ytick.labelsize"] = y_sz
    plt.savefig(f'../results/{title}.png', bbox_inches='tight',dpi=800, transparent = True)

# feature importance by drop_column
def drop_col_feat_imp(model, X_train, y_train, random_state = 42):
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    # list for storing feature importances
    importances = []
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis = 1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)
        importances.append(benchmark_score - drop_col_score)
    importances_df = imp_df(X_train.columns, importances)
    return importances_df

# feature importance by sklearn
def sklearn_feat_imp(rf, X_train):
    base_imp = imp_df(X_train.columns, rf.feature_importances_)
    var_imp_plot(base_imp, 'Default feature importance (sklearn)')

# feature importance by permutation
def permutation_feat_imp(rf, X_train, y_train):
    perm = PermutationImportance(rf, cv = None, refit = False, n_iter = 50).fit(X_train, y_train)
    perm_imp_eli5 = imp_df(X_train.columns, perm.feature_importances_)
    var_imp_plot(perm_imp_eli5, 'Permutation feature importance (eli5)')
    
def plot_AUC(rf, X_test, y_test):
    probs = rf.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='Random Forrest Classifier (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('../results/ROC.png', bbox_inches='tight',dpi=300, transparent = True)
    plt.close()

    precision, recall, thresholds = metrics.precision_recall_curve(y_test, preds)
    prc_auc = metrics.auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label='Random Forrest Classifier (area = %0.2f)' % prc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig('../results/PRC.png', bbox_inches='tight',dpi=300, transparent = True)

    print ('F1 scores:', metrics.f1_score(y_test, preds.round()))

    print ('Confusion Matrix')
    print (metrics.confusion_matrix(y_test, preds.round()))

def plot_decisionPath(estimator, X, class_name = ['Churned', 'Converted']):
    dot_data = tree.export_graphviz(estimator, out_file=None, 
                     feature_names=list(X.columns),  
                     class_names=class_name,  
                     filled=True, rounded=True,  
                     special_characters=True, precision = 0, impurity = False)  
    graph = pydotplus.graph_from_dot_data(dot_data)  
    graph.write_pdf("../results/RF_decision.pdf")
