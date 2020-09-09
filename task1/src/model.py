"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Author(s):
Yahav Bar
Shaked Haiman
Tzvi Cohen
===================================================
"""

import os
import re
from typing import Optional, Any

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from pandas.tseries.holiday import USFederalHolidayCalendar
from plotnine import ggplot, aes, geom_point, labs, geom_tile
from sklearn import metrics, linear_model
from sklearn.linear_model import LassoCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def get_initial_path() -> str:
    """
    Gets the inital path.
    :return: The initial path
    """
    # submissions should be relative to ./src (https://moodle2.cs.huji.ac.il/nu19/mod/forum/discuss.php?d=92765)
    return './'
    # return os.path.abspath('')


"""The path to the flights training data set."""
FLIGHTS_TRAIN_DATA_PATH = os.path.join(get_initial_path(), 'flight_data', 'train_data.csv')

"""The path to the weather training data set."""
WEATHER_TRAIN_DATA_PATH = os.path.join(get_initial_path(), 'all_weather_data', 'all_weather_data.csv')

"""The path to the flights test data set."""
FLIGHTS_TEST_DATA_PATH = os.path.join(get_initial_path(), 'flights_demo_test_file.csv')

"""The path to the regression model."""
REGRESSION_MODEL_OUTPUT_PATH = os.path.join(get_initial_path(), 'regression.pkl')

"""The path to the classifier model."""
CLASSIFIER_MODEL_OUTPUT_PATH = os.path.join(get_initial_path(), 'classifier.pkl')

"""The list of features the model has to have."""
FEATURES_MODEL_OUTPUT_PATH = os.path.join(get_initial_path(), 'features.pkl')

"""The list of labels the delay is classified by."""
CLASSIFICATION_LABELS_OUTPUT_PATH = os.path.join(get_initial_path(), 'labels.pkl')

"""The weather outliners."""
WEATHER_OUTLINERS = ['snow_in', 'precip_in', 'avg_wind_speed_kts']

"""A dummy => prefix map."""
FLIGHT_DUMMIES_MAP = {
    'DayOfWeek': 'weekday',
    'Reporting_Airline': 'airline',
    'Origin': 'origin',
    'Dest': 'dest'
}

"""A list of dropped flight features."""
DROPPED_FLIGHTS_FEATURES = ['Tail_Number', 'OriginCityName', 'OriginState', 'DestCityName',
                            'DestState', 'Flight_Number_Reporting_Airline', 'CRSElapsedTime', 'FlightDate']


# region Model Trainer

class FlightModelPreProcessor:
    """A holidays calendar, used to create a 'is holiday' feature.*"""
    _calendar = USFederalHolidayCalendar()

    """Factorized labels"""
    _factorized_data: Optional[DataFrame] = None

    @staticmethod
    def get_factorized_data():
        """
        Gets the factorized data.
        :return: The factorized data.
        """
        return FlightModelPreProcessor._factorized_data

    @staticmethod
    def apply_cleanup_pipeline(df: DataFrame, weather_file_path: Optional[str] = '',
                               drop_features: bool = False) -> DataFrame:
        """
        Apply the cleanup pipeline on the given data frame.
        :param df: The data frame.
        :param weather_file_path: The weather file.
        :param drop_features: True if we should optimize this model to drop unnecessary features, false otherwise.
        :return: The data frame, after being processed in the standard pipeline.
        """
        preprocess_pipeline = [
            FlightModelPreProcessor.fix_flight_date,
            lambda d: FlightModelPreProcessor.join_weather_db(d, weather_file_path)
            if weather_file_path else d,
            FlightModelPreProcessor.format_flight_time,
            FlightModelPreProcessor.add_holiday_information,
            FlightModelPreProcessor.add_dummies,
            FlightModelPreProcessor.add_flight_times_bins,
            FlightModelPreProcessor.add_is_same_state,
            lambda d: FlightModelPreProcessor.drop_features(d) if not drop_features else d,
        ]

        for pipeline_entry in preprocess_pipeline:
            df = pipeline_entry(df)

        df.fillna(0, inplace=True)
        return df

    @staticmethod
    def add_flight_times_bins(df: DataFrame) -> DataFrame:
        """
        Group the flight times into 2-hours bins.
        :param df: The data frame.
        :return: The modified data frame.
        """
        two_hours_bins = np.linspace(0, 2400, num=25)
        two_hours_labels = np.rint(np.linspace(0, 23, 24))

        df["DepartureBins"] = pd.cut(df['CRSDepTime'], bins=two_hours_bins, labels=two_hours_labels)
        df["ArrivalBins"] = pd.cut(df['CRSArrTime'], bins=two_hours_bins, labels=two_hours_labels)

        df.drop(['CRSDepTime', 'CRSArrTime'], axis=1)
        return df

    @staticmethod
    def drop_neglectable_entries(df: DataFrame) -> DataFrame:
        """
        Drops neglect-able data that might add some unnecessary noise to our data frame.
        :param df: The data frame.
        :return: The modified data frame.
        """
        df = df[abs(df["ArrDelay"] - np.mean(df["ArrDelay"])) < 3.5 * np.std(df["ArrDelay"])]
        df = df[~df["ArrDelay"].isna()]
        return df

    @staticmethod
    def factorize(df: DataFrame, field: str) -> DataFrame:
        """
        Factorize the given field.
        :param df: The data frame.
        :param field: The field to factorize.
        :return: The modified data frame.
        """
        FlightModelPreProcessor._factorized_data = df[field].factorize()
        df[field] = FlightModelPreProcessor._factorized_data[0]
        return df

    @staticmethod
    def add_dummies(df: DataFrame) -> DataFrame:
        """
        Adds dummies to categorize the data.
        :param df: The data frame.
        :return: The modified data frame.
        """
        return pd.get_dummies(df, columns=FLIGHT_DUMMIES_MAP.keys(), prefix=FLIGHT_DUMMIES_MAP.values())

    @staticmethod
    def drop_features(df: DataFrame) -> DataFrame:
        """
        Drop unnecessary features from the flights df.
        Adds dummies to categorize the data.
        :param df: The data frame.
        :return: The modified data frame.
        """
        return df.drop(DROPPED_FLIGHTS_FEATURES, axis=1)

    @staticmethod
    def format_flight_time(df: DataFrame) -> DataFrame:
        """
        Formats the flight date as a Pandas datetime.
        :param df: The data frame.
        :return: The modified data frame.
        """
        df['FlightDate'] = pd.to_datetime(df['FlightDate'], format="%d-%m-%y")
        return df

    @staticmethod
    def add_holiday_information(df: DataFrame) -> DataFrame:
        """
        Adds a "is holiday" feature.
        :param df: The data frame.
        :return: The modified data frame.
        """
        holidays = FlightModelPreProcessor._calendar.holidays(start=df['FlightDate'].min(), end=df['FlightDate'].max())
        df["IsHoliday"] = df["FlightDate"].isin(holidays)
        return df

    @staticmethod
    def add_is_same_state(df: DataFrame) -> DataFrame:
        """
        Adds a "is same state" feature.
        :param df: The data frame.
        :return: The modified data frame.
        """
        df["IsSameState"] = df["OriginState"] == df["DestState"]
        return df

    @staticmethod
    def join_weather_db(df: DataFrame, path_to_weather: str) -> DataFrame:
        """
        Joins the weather data frame with the flights db.
        :param df: The flights df.
        :param path_to_weather: The path to the weather df.
        :return: The joined df.
        """
        # Load
        weather_data_frame = FlightModelPreProcessor.load_weather_db(path_to_weather)

        # Join it with our data based on the given join columns (SQL like, lol)
        weather_origin_data = weather_data_frame.dropna().drop(columns=['station', 'FlightDate'])
        weather_dest_data = weather_origin_data.copy()
        weather_origin_data = weather_origin_data.add_suffix('_Origin')
        weather_dest_data = weather_dest_data.add_suffix('_Dest')
        weather_origin_data = pd.concat([weather_data_frame[['station', 'FlightDate']], weather_origin_data],
                                        axis=1).rename(columns={'station': 'Origin'})
        weather_dest_data = pd.concat([weather_data_frame[['station', 'FlightDate']], weather_dest_data],
                                      axis=1).rename(columns={'station': 'Dest'})

        return df \
            .merge(weather_origin_data, on=['Origin', 'FlightDate'], how='left') \
            .merge(weather_dest_data, on=['Dest', 'FlightDate'], how='left')

    @staticmethod
    def load_weather_db(path_to_weather: str):
        """
        Loads the weather df.
        :param path_to_weather: The path to the weather df.
        :return: The loaded weather df.
        """
        weather_df = pd.read_csv(path_to_weather,
                                 usecols=['station', 'day', 'max_temp_f', 'precip_in', 'avg_wind_speed_kts'],
                                 low_memory=False)

        # Fix the file contents
        weather_df = weather_df.rename(columns={'day': 'FlightDate'})
        weather_df.replace(to_replace=["None", "-100", "-99"], value=np.nan, inplace=True)
        weather_df.iloc[:, 2:] = weather_df.iloc[:, 2:].apply(pd.to_numeric)
        weather_df[weather_df['max_temp_f'].astype(float) > 130] = np.nan

        return weather_df

    @staticmethod
    def remove_weather_outliers(df: DataFrame) -> DataFrame:
        """
        Iterates and remove the weather outliners.
        :param df: The data frame.
        :return: The modified data frame.
        """
        for col in WEATHER_OUTLINERS:
            # Corrupted rows
            mean = df[((df[col] != 'None') & (df[col] != '-99') & ~df[col].isna())]
            [col].astype('float').mean()

            # Fix none values
            df[col] = df[col].astype('string')
            df[col].loc[df[col] == 'None'] = str(mean)
            df[col].loc[df[col].isna()] = str(mean)

            # Float conversation and removal of neglectable values
            df[col] = df[col].astype('float')
            df[col].loc[df[col].astype('float') < 0] = mean
            df[col].loc[abs(df[col].astype('float') - mean) < 3.5 * np.std(df[col].astype('float'))] = mean
            df[col][df['FlightDate'].str.contains('-0[3-9]-', na=False)] = 0

        return df

    @staticmethod
    def fix_flight_date(df: DataFrame) -> DataFrame:
        """
        Fixes the flight date entry.
        :param df: The data frame.
        :return: The modified data frame.
        """

        df['FlightDate'] = df['FlightDate'].apply(lambda d: re.sub(r'(\d\d)(\d\d)(-\d+-)(\d+)', r'\4\3\2', d))
        return df


class FlightModelTrainer:
    """
    A class that trains set of models that can detect flight delays.
    """

    """The used seed."""
    _random_seed: int

    def __init__(self, random_seed: int = 0):
        """
        Initialize the model trainer.
        :param random_seed: The trainer random seed.
        """
        # Setup the random seed
        self._random_seed = random_seed
        np.random.seed(random_seed)

    def train(self, compression_level=9):
        """
        Train the model.
        :param compression_level: The saved model compression level.
        """
        # Load the data
        flights_df = pd.read_csv(FLIGHTS_TRAIN_DATA_PATH, low_memory=False)

        # Merge
        X = flights_df.drop(["ArrDelay", "DelayFactor"], axis=1)
        y = flights_df[["ArrDelay", "DelayFactor"]]

        # Process the data
        print('prepare')
        processed_data = self._prepare_train_data(X, y)

        # Setup the training process
        y_train_regression = processed_data.loc[:, 'ArrDelay'].to_frame()
        y_train_classification = processed_data.loc[:, 'DelayFactor'].to_frame()
        x_train = processed_data.drop(['ArrDelay', 'DelayFactor'], axis=1)
        collected_features = x_train.columns.values.tolist()

        # Apply regression
        print('regression')
        lasso_regression = LassoCV(cv=5, random_state=self._random_seed).fit(x_train, y_train_regression.values.ravel())

        # Apply classification
        print('classification')
        classification_model = OneVsRestClassifier(DecisionTreeClassifier(max_depth=11))
        classification_model.fit(x_train, y_train_classification)

        # Save the results
        joblib.dump(lasso_regression, REGRESSION_MODEL_OUTPUT_PATH, compress=compression_level)
        joblib.dump(classification_model, CLASSIFIER_MODEL_OUTPUT_PATH, compress=compression_level)
        joblib.dump(collected_features, FEATURES_MODEL_OUTPUT_PATH, compress=compression_level)
        joblib.dump(FlightModelPreProcessor.get_factorized_data(), CLASSIFICATION_LABELS_OUTPUT_PATH,
                    compress=compression_level)

        print('Done.')

    def _do_train(self, x_train, x_validate, y_train, y_validate):
        """
        Trains the model using polynomial features.
        Unfortunately we couldn't finish to use this approach completely due to time reasons.
        We leave here the code to show our best efforts! :) :muscle: :muscle:

        :param x_train: The training data.
        :param x_validate: The validation data.
        :param y_train: The y train vector.
        :param y_validate: The y validation vector.
        """
        score_min = 10000
        for order in range(1, 3):
            for alpha in range(0, 20, 2):
                feature = PolynomialFeatures(degree=order)
                x_train = feature.fit_transform(x_train)
                validate_X = feature.fit_transform(x_validate)

                # ridge:
                ridge_model = linear_model.Ridge(alpha=alpha / 10, normalize=True)
                ridge_model.fit(x_train, y_train)
                result1 = ridge_model.predict(validate_X)

                # regression tree:
                reg = DecisionTreeRegressor(max_depth=alpha + 1)
                reg.fit(x_train, y_train)
                result2 = reg.predict(validate_X)

                score1 = metrics.mean_squared_error(result1, y_validate)
                score2 = metrics.mean_squared_error(result2, y_validate)

                if score1 < score_min:
                    score_min = score1

                print("n={} (a={}), MSE = {:<0.5}".format(order, alpha / 10, score1))
                print("tree: n={} (a={}), MSE = {:<0.5}".format(order, alpha + 1, score2))

    def _prepare_train_data(self, x: DataFrame, y: DataFrame,
                            drop_features: bool = False) -> DataFrame:
        """
        Prepare the data for training.
        :param x: The main data.
        :param y: The y vector.
        :param drop_features: True if we should optimize this model to drop unnecessary features, false otherwise.
        :return: The processed data.
        """
        df = x.join(y)
        df = FlightModelPreProcessor.drop_neglectable_entries(df)
        df = FlightModelPreProcessor.factorize(df, 'DelayFactor')
        return FlightModelPreProcessor.apply_cleanup_pipeline(df, WEATHER_TRAIN_DATA_PATH, drop_features)

    def visualize(self):
        """
        Visualize the data models.
        """
        # Load the data
        flights_df = pd.read_csv(FLIGHTS_TRAIN_DATA_PATH, low_memory=False)
        flights_df = flights_df.head(10000)

        # Merge
        X = flights_df.drop(["ArrDelay", "DelayFactor"], axis=1)
        y = flights_df[["ArrDelay", "DelayFactor"]]

        # Process the data
        print('prepare')
        df = self._prepare_train_data(X, y, True)

        # Declare an "is delayed" feature to use in the plots
        df["is_delayed"] = (df['DelayFactor'] != -1).astype(int)

        # Render a pair-plot with lot of comparative information
        print(sns.pairplot(df, vars=["CRSElapsedTime", "Distance", "Flight_Number_Reporting_Airline"], hue='is_delayed'))

        # Render specific features
        print((ggplot(df) +
              aes(x='Distance', y='ArrDelay', color='is_delayed') +
              geom_point() +
              labs(title=f"Distance V. Delay: ${round(df['ArrDelay'].corr(df['Distance']), 5)}$")))

        print(ggplot(df) +
              aes(x='CRSElapsedTime', y='ArrDelay', color='is_delayed') +
              geom_point() +
              labs(title=f"CRSElapsedTime V. Delay: ${round(df['ArrDelay'].corr(df['CRSElapsedTime']), 3)}$"))

        print(ggplot(df) +
              aes(x='DepartureBins', y='ArrDelay', color='is_delayed') +
              geom_point() +
              labs(title=f"DepartureBins V. Delay: ${round(df['ArrDelay'].corr(df['DepartureBins']), 3)}$"))

        print((ggplot(df) +
              aes(x='ArrivalBins', y='ArrDelay', color='is_delayed') +
              geom_point() +
              labs(title=f"ArrivalBin V. Delay: {round(df['ArrDelay'].corr(df['ArrivalBins']), 3)}")))

        # Create a tiles based indicators
        numeric_columns = pd.concat([df['ArrivalBins'], df['DepartureBins'], df['CRSElapsedTime'],
                                     df['Distance'], df['ArrDelay']], axis=1)
        correlation_matrix = numeric_columns.corr(method='pearson').round(2)
        correlation_matrix.index.name = 'variable2'
        correlation_matrix.reset_index(inplace=True)
        print(ggplot(pd.melt(correlation_matrix, id_vars=['variable2'])) +
              aes(x='variable', y='variable2', fill='value') +
              geom_tile() +
              labs(title='Numeric Columns Correlation'))


# endregion


# region Predictor

class FlightPredictor:
    """
    A class that uses a pre-made models to predicate flights delays.
    """

    """The path to the weather file."""
    weather_file_path: Optional[str]

    """A cached object that contains the used features, so that we can re-index the test data."""
    _features: Any

    """The classifier object."""
    _classifier: OneVsRestClassifier

    """The labels object."""
    _labels: Any

    def __init__(self, path_to_weather: str = ''):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        # Thaw the freeze'd models
        self._model = joblib.load(REGRESSION_MODEL_OUTPUT_PATH)
        self._classifier = joblib.load(CLASSIFIER_MODEL_OUTPUT_PATH)
        self._features = joblib.load(FEATURES_MODEL_OUTPUT_PATH)
        self._labels = joblib.load(CLASSIFICATION_LABELS_OUTPUT_PATH)

        self.weather_file_path = path_to_weather if path_to_weather != '' else None

    def predict(self, x):
        """
        Receives a pandas DataFrame of shape (m, 15) with m flight features, and predicts their
        delay at arrival and the main factor for the delay.
        @param x: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """

        # Prepare the data for evaluation
        df = self._prepare_test_data(x)

        # Regression
        regression = self._model.predict(df)

        # Classification
        classification = self._classifier.predict(df)
        FlightModelPreProcessor.get_factorized_data()

        pred_df = pd.DataFrame({
            'PredArrDelay': regression,
            'PredDelayFactor': classification
        })

        pred_df['PredDelayFactor'] = pred_df['PredDelayFactor'].apply(self.classify_by_label)
        pred_df.loc[pred_df['PredArrDelay'] <= 0, "PredDelayFactor"] = 'Nan'

        return pred_df

    def classify_by_label(self, row: int) -> str:
        """
        Gets the label associated with this row.
        :param row: The row.
        :return: The label.
        """
        try:
            if row != -1:
                return self._labels[1][row]

            return 'Nan'
        except KeyError:
            return 'Nan'

    def _prepare_test_data(self, X) -> DataFrame:
        """
        Prepare the test data for evaluation.
        :param X: The data frame values.
        :return:
        """
        joined_df = FlightModelPreProcessor.apply_cleanup_pipeline(X, self.weather_file_path)
        joined_df = joined_df.reindex(self._features, axis=1, fill_value=0)
        joined_df.fillna(0, inplace=True)
        return joined_df

# endregion
