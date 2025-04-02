from abc import ABC, abstractmethod
from drift_reports import base_drift_report
from drift_reports.image_drift_report import ImageDriftReport
from drift_reports.tabular_drift_report import TabularDriftReport
import alibi
import os
import pandas as pd
import cv2
import streamlit as st


class Usecase(ABC):
    """
    Base class for use cases.
    """
    @abstractmethod
    def get_report(self) -> base_drift_report:
        """
        Generate the report for the use case.

        Returns:
            str: Report as a string.
        """
        pass


class EeconeCase(Usecase):
    """
    Use case for mill sensors data.
    """

    def get_report(self, *args) -> base_drift_report:
        data_ref = 'data/tabular/mill_data_ref.csv'
        data_test = 'data/tabular/mill_data_test.csv'
        data_type = 'Tabular'
        dt_columns = ['timestamp']
        cat_columns = []
        report = TabularDriftReport(data_ref, data_test, dt_columns, cat_columns)

        return report
    

class IncomePredictionCase(Usecase):
    """
    Use case for income prediction data.
    """

    def get_report(self, *args) -> base_drift_report:
        adult = alibi.datasets.fetch_adult()
        X = adult.data
        n_ref = 10000
        n_test = 10000
        X_ref, X_t0, X_t1 = X[:n_ref], X[n_ref:n_ref + n_test], X[n_ref + n_test:n_ref + 2 * n_test]
        data_ref = pd.DataFrame(X_ref, columns=adult.feature_names)
        data_test = pd.DataFrame(X_t0, columns=adult.feature_names)
        data_type = 'Tabular'
        dt_columns = []
        cat_columns = [data_ref.columns[i] for i in list(adult.category_map.keys())]
        report = TabularDriftReport(data_ref, data_test, dt_columns, cat_columns)

        return report
    

class ArchimedesCase(Usecase):
    """
    Use case for Archimedes data.
    """

    def get_report(self, *args) -> base_drift_report:
        ref_slider_placeholder, test_slider_placeholder = args
        images = self.get_images('data/thermal_ironbow/')
        with ref_slider_placeholder:
            ref_start, ref_end = st.slider("Select Reference Image Range", 0, len(images), (0, 321), 1, key="ref_slider")
        with test_slider_placeholder:
            test_start, test_end = st.slider("Select Test Image Range", 0, len(images), (321, len(images)), 1, key="test_slider")
        data_ref = images[:321]
        data_test = images[321:]
        report = ImageDriftReport(data_ref, data_test)

        return report
    

    @staticmethod
    def get_images(folder):
        """Load images from a given folder and return as a list of numpy arrays."""
        images = []
        for filename in sorted(os.listdir(folder)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(folder, filename))
                if img is not None:
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
        return images