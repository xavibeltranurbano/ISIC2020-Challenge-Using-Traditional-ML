# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import warnings
class Training:
    def __init__(self,vec_train,vec_val, gt_train,gt_val,cv=2):
          self.cross_val=cv  # Number of folds
          warnings.filterwarnings("ignore", category=ConvergenceWarning)
          warnings.filterwarnings("ignore", category=FutureWarning)
          self.classifiers = [
              #('SVM', LinearSVC(dual=False)),
              ('XGB',xgb.XGBClassifier()),
              ('Random Forest', RandomForestClassifier(n_estimators=200)),
              # Suppress ConvergenceWarning
              ('Logistic Regression', LogisticRegression(max_iter=500))]
          self.X=pd.concat([vec_train,vec_val],ignore_index=True)
          self.y = pd.concat([gt_train, gt_val], ignore_index=True)

    def Voting(self,all_predictions,vec_gt_val):
        # We do Voting with all the previous models
        summed_vector = [sum(vector[i] for vector in all_predictions) for i in range(len(all_predictions[0]))]
        voting = np.where(np.array(summed_vector) > 1, 1, 0)
        accuracy = accuracy_score(np.array(vec_gt_val).ravel(), voting)
        return accuracy

    def fit(self):
          stratified_kf = StratifiedKFold(n_splits=self.cross_val, shuffle=True,random_state=42)  # You can adjust the random_state as needed
          results = {'XGB':[],
                     'Random Forest':[],
                     'Logistic Regression':[],
                     'Voting':[]}  # Dictionary to store results

          print("\n------------Results------------")
          n_fold=1
          for train_index, val_index in stratified_kf.split(self.X, self.y):
              print(f"\n·FOLD {n_fold}")
              X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
              y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]

              # Now you can use X_train, y_train for training and X_val, y_val for validation in each fold
              all_predictions = []
              for name, classifier in self.classifiers:
                  # Train the classifier
                  classifier.fit(X_train, np.array(y_train).ravel())

                  # Make predictions on the validation set
                  predictions = classifier.predict(X_val)
                  all_predictions.append(predictions)

                  # Calculate accuracy
                  accuracy = accuracy_score(np.array(y_val).ravel(), predictions)

                  # Store the accuracy in the results dictionary
                  print(f" {name} Accuracy:", accuracy)
                  results[name].append(accuracy)

              # We do Voting with all the previous models
              accuracy = self.Voting(all_predictions,y_val)
              print(f" Voting Accuracy:", accuracy)
              results['Voting'].append(accuracy)
              n_fold+=1

          print("\n------------Mean Results CV------------")
          for name, vec in results.items():
              print(f"{name}: {np.mean(vec)}")

if __name__ == "__main__":
    pyramid_level=256
    vec_features_train = pd.read_csv(
        f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/features_train_{pyramid_level}x{pyramid_level}.csv')  # Set index=False to exclude the index column
    vec_gt_train = pd.read_csv(
        f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/gt_train_{pyramid_level}x{pyramid_level}.csv')  # Set index=False to exclude the index column

    vec_features_val = pd.read_csv(
        f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/features_val_{pyramid_level}x{pyramid_level}.csv')  # Set index=False to exclude the index column
    vec_gt_val = pd.read_csv(
        f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/gt_val_{pyramid_level}x{pyramid_level}.csv')  # Set index=False to exclude the index column

    training=Training(vec_features_train,vec_features_val,vec_gt_train,vec_gt_val,cv=5)
    training.fit()