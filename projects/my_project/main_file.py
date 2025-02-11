import io 
import numpy as np 
import os 
import pandas as pd 
import requests 
import zipfile 
from pathlib import Path 

from datetime import datetime
from sklearn import datasets, ensemble 

from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import TargetDriftPreset
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset.regression_performance import RegressionPreset 
from evidently.test_suite import TestSuite
from evidently.test_preset import DataQualityTestPreset, DataStabilityTestPreset
from evidently.tests import *

from evidently.metrics import (
    RegressionQualityMetric,
    RegressionPredictedVsActualScatter,
    RegressionPredictedVsActualPlot,
    RegressionErrorPlot,
    RegressionAbsPercentageErrorPlot,
    RegressionErrorDistribution,
    RegressionErrorNormality,
    RegressionTopErrorMetric,
    RegressionErrorBiasTable,
    
    DatasetSummaryMetric,
    ColumnSummaryMetric,
    DatasetMissingValuesMetric,
    DatasetCorrelationsMetric
)


# Load data 
raw_data = pd.read_csv("data/hour.csv", header=0, sep=',', parse_dates=['dteday'], index_col='dteday') 
raw_data.head() 

# Get weeks number 
days = len(raw_data.index.unique())
weeks = days / 7

print(f'days = {days}; weeks = {weeks}')

# Config Regression Model 
REF_MONTH_START = '2011-01-01'
REF_MONTH_END = '2011-01-28'

CUR_MONTH_START = '2011-01-29'
CUR_MONTH_END = '2011-02-28'

CUR_WEEK_START = '2011-02-12'
CUR_WEEK_END = '2011-02-18'

target = 'cnt'
prediction = 'prediction'
numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'weekday']
categorical_features = ['season', 'holiday', 'workingday']

reports_dir = Path('reports') / f'{CUR_WEEK_START}_{CUR_WEEK_END}'
reports_dir.mkdir(exist_ok=True)

reference = raw_data.loc[REF_MONTH_START:REF_MONTH_END]
current = raw_data.loc[CUR_MONTH_START:CUR_MONTH_END]

regressor = ensemble.RandomForestRegressor(random_state = 0, n_estimators = 50)
regressor.fit(reference[numerical_features + categorical_features], reference[target])

ref_prediction = regressor.predict(reference[numerical_features + categorical_features])
current_prediction = regressor.predict(current[numerical_features + categorical_features])

reference['prediction'] = ref_prediction
current['prediction'] = current_prediction

# Model Monitoring 
column_mapping = ColumnMapping(
    target=target,
    prediction=prediction,
    numerical_features=numerical_features,
    categorical_features=categorical_features)

# Model Performance 
regression_performance_report = Report(metrics=[RegressionPreset()])
regression_performance_report.run(
    reference_data=reference,
    current_data=current.loc[CUR_WEEK_START:CUR_WEEK_END],
    column_mapping=column_mapping
)

model_performance_report_path = reports_dir / 'model_performance.html'
regression_performance_report.save_html(model_performance_report_path)

# Target Drift 
target_drift_report = Report(metrics=[TargetDriftPreset()])
target_drift_report.run(
    reference_data=reference,
    current_data=current.loc[CUR_WEEK_START:CUR_WEEK_END],
    column_mapping=column_mapping
)

target_drift_report_path = reports_dir / 'target_drift.html'
target_drift_report.save_html(target_drift_report_path)

# Data Drift 
column_mapping = ColumnMapping()
column_mapping.numerical_features = numerical_features

data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(
    reference_data=reference,
    current_data=current.loc[CUR_WEEK_START:CUR_WEEK_END],
    column_mapping=column_mapping
)

data_drift_report_path = reports_dir / 'data_drift.html'
data_drift_report.save_html(data_drift_report_path)

# Data Quality 
# by using the include_tests we call upon the test suit preset. 
data_quality_report = Report([DataQualityPreset()])
data_quality_report.run(
    reference_data=reference,
    current_data=current.loc[CUR_WEEK_START:CUR_WEEK_END],
    column_mapping=column_mapping
)

data_quality_report_path = reports_dir / 'data_quality.html'
data_quality_report.save_html(data_quality_report_path)

# Test Suite for Data Quality 
data_quality = TestSuite(tests=[
    DataQualityTestPreset(),
])

data_quality.run(reference_data=reference, 
                   current_data=current.loc[CUR_WEEK_START:CUR_WEEK_END])
data_quality_report_path = reports_dir / 'test_data_quality.html'
data_quality.save_html(data_quality_report_path)

# Custom Test set dataset-level 
data_drift_suite = TestSuite(tests=[
    TestShareOfDriftedColumns(),
    TestNumberOfEmptyRows(),
])
data_drift_suite.run(reference_data=reference,current_data=current.loc[CUR_WEEK_START:CUR_WEEK_END])
data_quality_report_path = reports_dir / 'dataset_level_quality.html'
data_drift_suite.save_html(data_quality_report_path)
# Custom Test set columns-level
feature_suite = TestSuite(tests=[
    TestColumnShareOfMissingValues(column_name='workingday'),
    TestColumnDrift(column_name='weathersit'),
    TestMeanInNSigmas(column_name='workingday')
])

feature_suite.run(reference_data=reference,current_data=current.loc[CUR_WEEK_START:CUR_WEEK_END])
data_quality_report_path = reports_dir / 'columns_level_quality.html'
feature_suite.save_html(data_quality_report_path)