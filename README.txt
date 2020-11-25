file list:

1. main:
  trains the model, creates a predictor and prints the results.

2. model:
  contains the following classes:

   FlightModelPreProcessor - contains static methods to pre-processing the data.
   * fixes the flightDate entries and format it to datetime object
   * merges the weather data with the main data
   * adds the USA holiday feature to the data
   * adds the is_same_state feature to the data.
   * converts some of the features to categorial using dummy-values.
   * converts the flight depeture and arrival to time bins.
   * drops some feature we decided to exclude.

   FlightModelVisualizer - creates some visual graphs.

   FlightModelTrainer - trains set of models to predict flight delays.
   * loads the data
   * split the data to train, validate, test
   * pre- processor the train data
   * fits linear regression to predict if there will be a delay and how long it will be.
   * fits one vs rest classifier to predict the reason for the delay.
   * saves the results.

   FlightPredictor - uses a pre-made models to predict flight delays
   * contains the trained models.
   * apply the pre-processor stage on the test data
   * predicts the flight delays and reasons
   * calculate the L2 and L0-1 losses
   
3. all_weather_data directory and csv file are missing and are optional for the model. One may add weather data to improve the model.

Tzvi Cohen, Yahav Bar, Shaked Heiman.
