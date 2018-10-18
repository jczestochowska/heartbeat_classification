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
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper, class_names=self.class_names)
        ret_exp.predict_proba = yss[0]
        print("Explaining")
        explained = []
        for label in labels:
            explained.append(self.base.explain_instance_with_data(data, yss, distances, label, num_features,
                                                                  feature_selection=self.feature_selection))
        return explained

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
        sample = np.random.randint(1, num_slices + 1, num_samples - 1)
        data = np.ones((num_samples, num_slices))
        features_range = range(num_slices)
        inverse_data = [time_series.copy()]

        for i, size in enumerate(sample, start=1):
            if i % 100 == 0:
                print("Now processing {} feature from timeseries".format(i))
            inactive = np.random.choice(features_range, size, replace=False)
            # set inactive slice to mean of training_set
            data[i, inactive] = 0
            tmp_series = time_series.copy()
            for i, inact in enumerate(inactive, start=1):
                index = inact * values_per_slice
                # use mean as inactive
                tmp_series[index:(index + values_per_slice)] = np.mean(
                    training_set[:, index:(index + values_per_slice)])
            inverse_data.append(tmp_series)
        labels = classifier_fn(inverse_data)
        distances = distance_fn(data)
        return data, labels, distances