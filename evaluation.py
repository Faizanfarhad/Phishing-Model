from sklearn.metrics import (f1_score,precision_score
                            ,recall_score,
                            accuracy_score,
                            confusion_matrix,
)

class Evaluation:
    def __init__(self,y_pred,y_test):
        super().__init__()
        self.y_pred = y_pred
        self.y_test = y_test
    
    def logistic_model_scores(self):
        ''' return Scores:- Accuracy , Precision , Recall , F1Score , Confusion Matrix'''
        accuracy =  accuracy_score(y_true=self.y_test,y_pred=self.y_pred)
        precision = precision_score(y_true=self.y_test,y_pred=self.y_pred)
        recall = recall_score(y_true=self.y_test,y_pred=self.y_pred)
        f1 = f1_score(y_true=self.y_test,y_pred=self.y_pred)
        c_matrix = confusion_matrix(y_true=self.y_test,y_pred=self.y_pred)
        return accuracy,precision,recall,f1,c_matrix
