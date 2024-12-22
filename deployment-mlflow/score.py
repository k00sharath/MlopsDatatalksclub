
import sys

import uuid
import sys
import pandas as pd
import mlflow

from datetime import datetime

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("green-taxi-duration")
def generate_uuids(n):

    ride_ids = []
    for i in range(n):

        ride_ids.append(str(uuid.uuid4()))

    return ride_ids 

def read_dataframe(filename):

    df = pd.read_parquet(filename)
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds()/60
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
    df['ride_id'] = generate_uuids(len(df))

    return df

def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts

def load_model(run_id):
    logged_model = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(logged_model)
    return model

def save_results(df, y_pred, run_id, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id

    df_result.to_parquet(output_file, index=False)

def apply_model(input_file, run_id, output_file):
    

    
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

   
    model = load_model(run_id)

   
    y_pred = model.predict(dicts)


    save_results(df, y_pred, run_id, output_file)
    return output_file

def get_paths(year, month,taxi_type, run_id):
    

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year}-{month}.parquet'
    output_file = f'./output/nyc-duration-pred_{taxi_type}_{year:04d}_{month:02d}_{run_id}.parquet'

    return input_file, output_file

def ride_duration_prediction(
        taxi_type: str,
        run_id: str,
       year: str = '2024',
       month: str = '09'):
    
    
    input_file, output_file = get_paths(year, month, taxi_type, run_id)

    apply_model(
        input_file=input_file,
        run_id=run_id,
        output_file=output_file
    )

def run():
    taxi_type = sys.argv[1] 
    year = int(sys.argv[2]) 
    month = int(sys.argv[3])

    run_id = sys.argv[4] 

    ride_duration_prediction(
        taxi_type=taxi_type,
        run_id=run_id,
        year = year,
        month = month
    )


if __name__ == '__main__':
    run()
