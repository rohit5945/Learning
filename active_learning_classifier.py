import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# Generate synthetic data for demonstration
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# Split the data into labeled and unlabeled sets
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size=0.9, random_state=42)

# Train an initial classifier on the labeled data
classifier = SVC()
classifier.fit(X_labeled, y_labeled)

# Initial accuracy on the test set
X_test, y_test = make_classification(n_samples=500, n_features=20, n_informative=15, n_redundant=5, random_state=42)
initial_accuracy = accuracy_score(y_test, classifier.predict(X_test))
print(f"Initial Accuracy: {initial_accuracy:.2f}")

# Active learning loop
query_limit = 100  # Set a limit on the number of queries
query_batch_size = 10  # Number of instances to query in each iteration

for query_iter in range(query_limit):
    # Predict probabilities for the unlabeled data
    y_prob_unlabeled = classifier.decision_function(X_unlabeled)
    
    # Choose instances with the highest uncertainty (least confident predictions)
    uncertain_instances = np.argsort(np.abs(y_prob_unlabeled.max(axis=1)))[:query_batch_size]
    
    # Query the true labels for the uncertain instances
    queried_labels = y_unlabeled[uncertain_instances]
    
    # Update the labeled data with the queried instances
    X_labeled = np.concatenate([X_labeled, X_unlabeled[uncertain_instances]])
    y_labeled = np.concatenate([y_labeled, queried_labels])
    
    # Remove the queried instances from the unlabeled data
    X_unlabeled = np.delete(X_unlabeled, uncertain_instances, axis=0)
    y_unlabeled = np.delete(y_unlabeled, uncertain_instances)
    
    # Shuffle the labeled data to maintain randomness
    X_labeled, y_labeled = shuffle(X_labeled, y_labeled, random_state=42)
    
    # Retrain the classifier on the updated labeled data
    classifier.fit(X_labeled, y_labeled)
    
    # Evaluate the model on the test set
    accuracy = accuracy_score(y_test, classifier.predict(X_test))
    
    print(f"Iteration {query_iter + 1}/{query_limit}, Accuracy: {accuracy:.2f}")

    # Stop active learning if the model reaches a satisfactory accuracy
    if accuracy >= 0.95:
        print("Satisfactory accuracy reached. Stopping active learning.")
        break
