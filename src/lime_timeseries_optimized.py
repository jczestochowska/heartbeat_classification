import math

import numpy as np
import sklearn
from lime import explanation
from lime import lime_base


class LimeTimeSeriesExplanation(object):
    """Explains time series classifiers."""

    def __init__(self,
                 kernel_width=25,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 ):
        """Init function.
        Args:
            kernel_width: kernel width for the exponential kernel
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
            classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
        """

        # exponential kernel
        def kernel(d): return np.squeeze(np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2)))

        self.base = lime_base.LimeBase(kernel, verbose)
        self.class_names = class_names
        self.feature_selection = feature_selection

    def explain_instance(self,
                         timeseries,
                         classifier_fn,
                         training_set,
                         num_slices,
                         labels=(1, 0),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='cosine',
                         model_regressor=None,
                         replacement_method='mean'):
        """Generates explanations for a prediction.
        Args:
            time_series: Time Series to be explained.
            classifier_fn: classifier prediction probability function
            num_slices: Defines into how many slices the series will be split up
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
            the K labels with highest prediction probabilities, where K is
            this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
            defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
       """
        domain_mapper = explanation.DomainMapper()
        data, yss, distances = self.__data_labels_distances(timeseries, classifier_fn, num_samples, num_slices,
                                                            training_set, replacement_method)
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        explanation_ = explanation.Explanation(domain_mapper=domain_mapper, class_names=self.class_names)
        explanation_.predict_proba = yss[0]
        for label in labels:
            explanation_.local_exp[label] = \
                self.base.explain_instance_with_data(data, yss, distances, label, num_features,
                                                     feature_selection=self.feature_selection)[1]
        return explanation_

    @classmethod
    def __data_labels_distances(cls,
                                time_series,
                                classifier_fn,
                                num_samples,
                                num_slices,
                                training_set,
                                replacement_method='mean'):
        """Generates a neighborhood around a prediction.
        Generates neighborhood data by randomly removing words from
        the instance, and predicting with the classifier. Uses cosine distance
        to compute distances between original and perturbed instances.
        Args:
            time_series: Time Series to be explained.
            classifier_fn: classifier prediction probability function, which
                takes a time series and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear model
            num_slices: how many slices the time series will be split into for discretization.
            training_set: set of which the mean will be computed to use as 'inactive' values.
            replacement_method: Defines how individual slice will be deactivated (can be 'mean', 'total_mean', 'noise')
        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of tokens in indexed_string. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: cosine distance between the original instance and
                    each perturbed instance (computed in the binary 'data'
                    matrix), times 100.
        """

        def distance_fn(x):
            original_point = x[0, :].reshape(1, -1)
            distances = sklearn.metrics.pairwise_distances(x, original_point)
            return distances * 100

        # split time_series into slices
        values_per_slice = math.ceil(len(time_series) / num_slices)
        # compute randomly how many slices will be switched off
        sample = np.random.randint(1, num_slices, num_samples)
        # sparse matrix as interpretable features, each slice represents a feature
        sparse_matrix = np.ones((num_samples, num_slices))
        features_range = range(num_slices)
        petrubed_data = [time_series.copy()]
        inactive_features_in_dense_data = [np.random.choice(features_range, size, replace=False) for size in sample]

        for i in range(len(inactive_features_in_dense_data)):
            sparse_matrix[i, inactive_features_in_dense_data[i]] = 0

        means, stds = [], []
        for index in range(0, len(time_series), values_per_slice):
            feature = training_set[:, index:(index + values_per_slice)]
            mean = np.mean(feature)
            std = np.std(feature)
            means.append(mean)
            stds.append(std)
        means = np.array(means)
        stds = np.array(stds)

        for inactive in inactive_features_in_dense_data:
            petrubed_sample = time_series.copy()
            for index in inactive:
                petrubed_sample[index * values_per_slice:(index * values_per_slice) + values_per_slice] = \
                    np.random.normal(0, 1) * stds[index] + means[index]
            petrubed_data.append(petrubed_sample)

        labels = classifier_fn(petrubed_data)
        # add original point to sparse matrix
        sparse_matrix = np.insert(sparse_matrix, 0, np.ones((1, num_slices)), axis=0)
        distances = distance_fn(sparse_matrix)
        return sparse_matrix, labels, distances
