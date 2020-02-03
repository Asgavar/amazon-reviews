import functools

import numpy as np
import pandas as pd
import sklearn.feature_extraction.text
import sklearn.metrics


@functools.lru_cache
def load_dataset(sampling_method, vectorization, preprocessing):
    vectorizers = {
        'count': {
            None: sklearn.feature_extraction.text.CountVectorizer(),
            'stop_words': sklearn.feature_extraction.text.CountVectorizer(stop_words='english')
        },
        'binary': {
            None: sklearn.feature_extraction.text.CountVectorizer(binary=True),
            'stop_words': sklearn.feature_extraction.text.CountVectorizer(binary=True, stop_words='english'),
        },
        'tf_idf': {
            None: sklearn.feature_extraction.text.TfidfVectorizer(),
            'stop_words': sklearn.feature_extraction.text.TfidfVectorizer(stop_words='english'),
        }
    }
    
    vectorizer = vectorizers[vectorization][preprocessing]
    
    filenames = {
        'random_downsampling': ('downsampled_train.csv', 'full_test.csv'),
        'full': ('full_train.csv', 'full_test.csv'),
    }
    
    train_name, test_name = filenames[sampling_method]
    
    train = pd.read_csv(train_name, header=0, index_col=0)
    test = pd.read_csv(test_name, header=0, index_col=0)
    
    # print(train['reviewText'].values)
    # print(train['reviewText'].index)
    # print(train.dtypes)

    train_as_vector = vectorizer.fit_transform(train['reviewText'].values)
    test_as_vector = vectorizer.transform(test['reviewText'].values)
    
    return train_as_vector, train['overall'].values, test_as_vector, test['overall'].values


# load_dataset('random_downsampling', 'count', None)


def get_score(classifier, test, test_targets):
    return sklearn.metrics.balanced_accuracy_score(test_targets, classifier.predict(test))


def display_confusion_matrices(classifier, test, test_targets):
    print(sklearn.metrics.confusion_matrix(
        test_targets, classifier.predict(test), normalize='true'))
    sklearn.metrics.plot_confusion_matrix(
        classifier, test, test_targets, normalize='pred', cmap='Oranges')
    

def display_score(classifier, test, test_targets):
    print(f'SCORE: {get_score(classifier, test, test_targets)}')


def display_classifier_performance(classifier, test, test_targets):
    display_score(classifier, test, test_targets)
    display_confusion_matrices(classifier, test, test_targets)