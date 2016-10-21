# Question 1
# Joe Silverstein

import sklearn
from sklearn import cross_validation, datasets, metrics
from sklearn.linear_model import SGDClassifier
from sigopt import Connection
import numpy

moons = sklearn.datasets.make_moons(n_samples=1000)
X = moons[0]
y = moons[1]

conn = Connection(client_token="XPNSTEJSFNPXSNJKUDVKKKLVYZHMQGKPLSJKSQUIRMNGWAZH")

# Had to play around with the alpha bounds a bit.
# The optimal alpha was 0.01 for an accuracy improvement of 6% in the best experiment, which is
# a boundary case.
# I tried lowering the min alpha in subsequent experiments, but I couldn't ever get close to 6%
# so I might have just got lucky in experiment 8747. Either way, that was the best run.
experiment = conn.experiments().create(
    name='SGD Classifier (Python)',
    parameters=[
        {
            'bounds': {
                'max': 0.1,
                'min': 0.01
            },
            'name': u'alpha',
            'type': 'double'
        },
        {
            'categorical_values': [
                {
                    'name': u'hinge'
                },
                {
                    'name': u'log'
                },
                {
                    'name': u'modified_huber'
                },
                {
                    'name': u'squared_hinge'
                },
                {
                    'name': u'perceptron'
                }
            ],
            'name': u'loss',
            'type': 'categorical'
        }
    ],
)
print("Created experiment: https://sigopt.com/experiment/" + experiment.id)

def evaluate_model(assignments, X, y):
    classifier = SGDClassifier(loss = assignments['loss'], alpha = assignments['alpha'])
    cv_accuracies = cross_validation.cross_val_score(
        classifier, X, y, cv = 5, scoring = 'accuracy')
    return (numpy.mean(cv_accuracies), numpy.std(cv_accuracies))

'''
Note: 
I used the accuracy score as the validation loss because it measures the percentage of test data that was correctly predicted in each hold-out sample. This seems natural for a classification problem when I don't have additional contextual information about whether we care more about false positives or false negatives, for example.
'''

# Run the optimization loop
for _ in range(15):
    suggestion = conn.experiments(experiment.id).suggestions().create()
    (value, std) = evaluate_model(suggestion.assignments, X, y)
    conn.experiments(experiment.id).observations().create(
        suggestion = suggestion.id, value = value, value_stddev = std)

# Re-fetch the experiment to get the best observed value and assignments
experiment = conn.experiments(experiment.id).fetch()
best_assignments = experiment.progress.best_observation.assignments

# Fit the classifier on the best assignments and train on all available data
sgd = SGDClassifier(loss = best_assignments['loss'], alpha = best_assignments['alpha'])
sgd.fit(X,y)



