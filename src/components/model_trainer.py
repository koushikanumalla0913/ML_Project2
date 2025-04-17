import os, sys
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define k values to evaluate
            k_values = [1, 5, 10, 20, 50, 100, 200]
            best_k = 1
            best_score = 0

            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
                mean_score = scores.mean()
                logging.info(f"k={k}: Cross-validation accuracy={mean_score:.4f}")
                if mean_score > best_score:
                    best_score = mean_score
                    best_k = k

            logging.info(f"Best k value: {best_k} with CV accuracy: {best_score:.4f}")

            if best_score < 0.6:
                raise CustomException("No suitable KNN model found")

            
            best_model = KNeighborsClassifier(n_neighbors=best_k)
            best_model.fit(X_train, y_train)

         
            save_object(self.config.trained_model_file_path, best_model)

            
            y_pred = best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            
            logging.info(f"Confusion Matrix for k={best_k}:\n{cm}")

            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
            plt.title(f"Confusion Matrix for k={best_k}")
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.show()
            final_accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Final test accuracy: {final_accuracy:.4f}")

            return final_accuracy

        except Exception as e:
            raise CustomException(e, sys)