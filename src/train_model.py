import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def train_and_evaluate_model():
    '''
    Train Default Loan Prediction model_features

    Also, runs accuracy evaluation on the model using a 20 percent validation set of the original data

    Saves trained model in:
        models/load_default_prediction_model.p
    '''
    features_df = pd.read_csv("data/model_features.csv")
    labels_df = pd.read_csv("data/model_labels.csv")

    features = features_df.ix[:,1:].as_matrix()
    labels = labels_df.ix[:,1].as_matrix()

    # Split into Train, Validation Set
    X_train, X_validation, y_train, y_validation = train_test_split(features, labels, test_size=0.2)

    print 'Training Model'
    model = RandomForestClassifier()  # default parameters for now just to save space / serve as proof of concept
    model.fit(X_train, y_train)
    print "Training Complete"

    print "Evaluation:"
    y_valiation_predictions = model.predict(X_validation)
    print "  - Positive Fraction in Data: " + str(float(sum(labels)) / len(labels))
    print "  - Validation Accuracy: " + str(accuracy_score(y_validation, y_valiation_predictions))

    print 'Saving Modeling File'
    with open('models/load_default_prediction_model.p', 'wb') as model_file:
        model_file.write(pickle.dumps(model))

if __name__ == '__main__':
    train_and_evaluate_model()