# -----------------------------------------------------------------------------
# Training Class
# Author: Xavier Beltran Urbano
# Date Created: 31-10-2023
# -----------------------------------------------------------------------------

# Import necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

class Training:
    def __init__(self,vec_train,vec_val, gt_train,gt_val,type_training,cv=2):
          """
            Training class with training and validation data, classifiers, and other parameters.
            Args:
                vec_train (pd.DataFrame): Training data features.
                vec_val (pd.DataFrame): Validation data features.
                gt_train (pd.DataFrame): Ground truth for training data.
                gt_val (pd.DataFrame): Ground truth for validation data.
                type_training (str): Type of training (e.g., "Binary", "Multiclass"). Determines the training approach.
                cv (int, optional): Number of cross-validation folds. Defaults to 2.
          """
          self.cross_val=cv  # Number of folds
          warnings.filterwarnings("ignore", category=ConvergenceWarning)
          warnings.filterwarnings("ignore", category=FutureWarning)
          self.classifiers = [
              ('XGB',xgb.XGBClassifier(learning_rate=0.5,n_estimators=300,max_depth=5,subsample=1.0,colsample_bytree=0.8,gamma=0)),
              ('Random Forest',RandomForestClassifier(max_depth=30,min_samples_leaf=1, min_samples_split=5,n_estimators=300))]
          self.X=pd.concat([vec_train,vec_val],ignore_index=True)
          self.y = pd.concat([gt_train, gt_val], ignore_index=True)
          self.thresholds=[] # We will store all the youden index
          self.type_training=type_training # Binary or Multiclass

    def Voting(self,all_predictions,vec_gt_val):
        # We do Voting with all the previous models
        summed_vector = [sum(vector[i] for vector in all_predictions) for i in range(len(all_predictions[0]))]

        # Put 1 if the sum of the previous predictions is >1. Otherwise, put 0
        voting = np.where(np.array(summed_vector) > 1, 1, 0)

        # Compute the accuracy of the voting
        accuracy = accuracy_score(np.array(vec_gt_val).ravel(), voting)
        return accuracy

    def computeMetrics(self,predictions,y_val, name_classifier,results):
        # Calculate accuracy
        accuracy = accuracy_score(np.array(y_val).ravel(), predictions)

        # Store the accuracy in the results dictionary
        print(f" {name_classifier} Accuracy:", accuracy)
        results[name_classifier].append(accuracy)

        # Calculate the confusion matrix
        confusion = confusion_matrix(np.array(y_val).ravel(), predictions)

        # Print the confusion matrix
        print("Confusion Matrix:")
        print(confusion)

        # Generate the entire classification report
        report = classification_report(np.array(y_val).ravel(), predictions)

        # Print the classification report
        print("\nClassification Report:")
        print(report)
        return results

    def plotROC(self,fpr,tpr,auc_value):
        # Plotting the ROC curve
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_value:.2f}')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
        plt.savefig("/Users/xavibeltranurbano/PycharmProjects/ISIC-Challenge-A-Conventional-Skin-Lesion-Classification-Approach/Binary/ROC_Curve.png")

    def computeYoudenIndex(self,prob_predictions,y_val,plot_ROC):
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_val, prob_predictions)

        # Calculate AUC (Area under the ROC Curve)
        auc_value = auc(fpr, tpr)

        # Plot ROC
        if plot_ROC:
            self.plotROC(fpr,tpr,auc_value)

        # Calculate specificities
        specificities = 1 - fpr

        # Compute Youden Index for each threshold
        youden_indices = tpr + specificities - 1

        # Identify the threshold that maximizes the Youden Index
        optimal_threshold = thresholds[np.argmax(youden_indices)]
        return optimal_threshold

    def binarizeProbabilities(self, prob_predictions,y_val):
        # Get the best threshold computing the youden index
        threshold = self.computeYoudenIndex(prob_predictions,y_val, plot_ROC=False)
        self.thresholds.append(threshold)

        # Binarize probabilities using youden index
        print(f"Threshold: {threshold}")
        predictions = np.where(prob_predictions > threshold, 1, 0)

        return predictions

    def train_classifier(self,classifier,X_train,y_train,X_val,y_val,all_predictions,name, results):
        # Train the classifier
        classifier.fit(X_train, np.array(y_train).ravel())

        # Make predictions on the validation set
        prob_predictions = classifier.predict_proba(X_val)[:, 1]

        # Binarize probabilities
        predictions = self.binarizeProbabilities(prob_predictions, y_val)
        all_predictions.append(predictions)

        # Compute metrics
        results = self.computeMetrics(predictions, y_val, name, results)
        return all_predictions, results


    def fit(self):
          stratified_kf = StratifiedKFold(n_splits=self.cross_val, shuffle=True,random_state=42)  # You can adjust the random_state as needed
          results = {'XGB':[], 'Random Forest':[], 'Voting':[]}  # Dictionary to store results
          print("\n------------Results------------")
          n_fold=1
          for train_index, val_index in stratified_kf.split(self.X, self.y):
              print(f"\n·FOLD {n_fold}")
              # Extract the samples of this fold
              X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
              y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]

              all_predictions = []
              for name, classifier in self.classifiers:
                  # Train the classifier and compute metrics
                  all_predictions,results=self.train_classifier(classifier,X_train,y_train,X_val,y_val,all_predictions,name, results)

              # We do Voting with all the previous models
              accuracy = self.Voting(all_predictions,y_val)
              print(f" Voting Accuracy:", accuracy)
              results['Voting'].append(accuracy)
              n_fold+=1

          # Print the results of the CV
          print("\n------------Mean Results CV------------")
          for name, vec in results.items():
              print(f"{name}: {np.mean(vec)}")

    def predict_test(self, test):
        # Select the XGB classifier from the classifiers list
        classifier = [clf for name, clf in self.classifiers if name == 'XGB'][0]

        # Train the classifier using the entire dataset
        classifier.fit(self.X, np.array(self.y).ravel())

        # Make predictions on the test set
        prob_predictions = classifier.predict_proba(test)[:, 1]

        # Binarize probabilities using the mean threshold (Youden Index) from cross-validation
        predictions = np.where(prob_predictions > np.mean(self.thresholds), 1, 0)

        # Save predictions to a CSV file
        df_predictions = pd.DataFrame(predictions, columns=['Predictions Test'])
        df_predictions.to_csv(f'/Users/xavibeltranurbano/PycharmProjects/ISIC-Challenge-A-Conventional-Skin-Lesion-Classification-Approach/Binary/{self.type_training}_test_predictions.csv', index=False)


if __name__ == "__main__":
    # Read features
    vec_features_train = pd.read_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/new_features_train_256x256.csv')  # Set index=False to exclude the index column
    vec_gt_train = pd.read_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/new_gt_train_256x256.csv')  # Set index=False to exclude the index column
    vec_features_val = pd.read_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/new_features_val_256x256.csv')  # Set index=False to exclude the index column
    vec_gt_val = pd.read_csv(f'/Users/xavibeltranurbano/Desktop/MAIA/GIRONA/CAD/MACHINE LEARNING/BINARY/features/new_gt_val_256x256.csv')  # Set index=False to exclude the index column

    # Start training the model
    training=Training(vec_features_train,vec_features_val,vec_gt_train,vec_gt_val, 'Binary',cv=5)
    training.fit()
    training.predict_test(vec_features_val)
