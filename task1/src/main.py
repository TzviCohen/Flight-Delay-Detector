import sys
import pandas as pd
from model import FlightPredictor, FlightModelTrainer, WEATHER_TRAIN_DATA_PATH, FLIGHTS_TEST_DATA_PATH

"""
Usage:
- to train the model: ./main.py train
- to visualize: ./main.py graphics
- to predict: ./main.py
"""

def main():
    """
    The program main entry point.
    :return: The exit status code.
    """
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        trainer = FlightModelTrainer()
        trainer.train()
        return 0

    if len(sys.argv) == 2 and sys.argv[1] == 'graphics':
        trainer = FlightModelTrainer()
        trainer.visualize()
        return 0

    predictor = FlightPredictor(path_to_weather=WEATHER_TRAIN_DATA_PATH)
    result = predictor.predict(pd.read_csv(FLIGHTS_TEST_DATA_PATH))
    print('result')
    print(result)
    # result.to_csv("out.csv")
    return 0


if __name__ == '__main__':
    sys.exit(main())
