from airflow import DAG
from airflow.operators import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime, timedelta
from airflow.contrib.sensors.file_sensor import FileSensor

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

#step 1, get train data
def get_training_data(**kwargs):
    df = pd.read_csv("~/excel/train_airflow.csv")
    kwargs['ti'].xcom_push(key='dataset', value=df)

#step 2, preprocess and model the data
def preprocessing_training(**kwargs):
    df = kwargs['ti'].xcom_pull(task_ids='get_training_data', key='dataset')
    X = df.drop(columns = 'deposit', axis = 1)
    y = df['deposit'].map({'yes':1, 'no':0})

    #simple data preprocessing
    numerical_features = X.select_dtypes(exclude = ['object']).columns.tolist()
    categorical_features = X.select_dtypes(include = ['object']).columns.tolist()

    preprocessor = make_column_transformer((StandardScaler(),numerical_features),\
                                           (OneHotEncoder(),categorical_features),\
                                           remainder ='passthrough')
                                           
    preprocessed_fitted = preprocessor.fit(X)
    preprocessed_X = preprocessed_fitted.transform(X)
    model = GradientBoostingClassifier()
    fitted = model.fit(preprocessed_X,y)
    score = fitted.score(preprocessed_X,y)
    kwargs['ti'].xcom_push(key='score', value=score)
    kwargs['ti'].xcom_push(key='preprocessed_fitted', value=preprocessed_fitted)
    kwargs['ti'].xcom_push(key='fitted', value=fitted)
        
#step 3 testing
def testing(**kwargs):
    df = pd.read_csv("~/excel/test_airflow.csv")
    X_test = df.drop(columns = 'deposit', axis = 1)
    y_test = df['deposit'].map({'yes':1, 'no':0})

    preprocessed_fitted = kwargs['ti'].xcom_pull(task_ids='preprocessing_training', key='preprocessed_fitted')
    preprocessed_X = preprocessed_fitted.transform(X_test)
    
    fitted = kwargs['ti'].xcom_pull(task_ids='preprocessing_training', key='fitted')
    test_score = fitted.score(preprocessed_X,y_test)

    kwargs['ti'].xcom_push(key='test_score', value=test_score)
    
#step 4 comparing the results
def branch(**kwargs):
    previous_score = kwargs['ti'].xcom_pull(task_ids='testing', key='test_score')
    
    if previous_score >=0.8:
        val = 'model_good_for_production'
    else:
        val = 'model_needs_retrain'
    return val

#Step 5a, ready to make prediction with current model 
def model_good_for_production(**kwargs):
    print('Current Model can be used to make prediction')
    
#Step 5b, retraining the modelling 
def model_needs_retrain(**kwargs):
    df = pd.read_csv("~/excel/test_airflow.csv")
    X_test = df.drop(columns = 'deposit', axis = 1)
    y_test = df['deposit'].map({'yes':1, 'no':0})

    #simple data preprocessing

    preprocessed_fitted = kwargs['ti'].xcom_pull(task_ids='preprocessing_training', key='preprocessed_fitted')
    preprocessed_X = preprocessed_fitted.transform(X_test)
    model = GradientBoostingClassifier()
    
 #the refitted model can now be used to fit new data moving forward
    re_fitted = model.fit(preprocessed_X,y)

    
    
##################################################################################
#create the ML_pipeline dag
default_args = {
    'owner': 'aaron',
    'depends_on_past': False,
    'start_date': datetime(2020,2,2),
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}
dag = DAG(
    'ML_pipeline_demo',
    catchup=False,
    default_args=default_args,
    schedule_interval="@once")

sensors = FileSensor(
    task_id = 'sensors', 
    filepath = "~/excel/train_airflow.csv",
    fs_conn_id = 'fs_default',
    poke_interval = 10,
    timeout = 150,
    soft_fail = True
)
    

t1 = PythonOperator(
    task_id='get_training_data',
    python_callable=get_training_data,
    provide_context=True,
    dag=dag
)

t2 = PythonOperator(
    task_id='preprocessing_training',
    python_callable=preprocessing_training,
    provide_context=True,
    dag=dag
)

t3 = PythonOperator(
    task_id='testing',
    python_callable=testing,
    provide_context=True,
    dag=dag
)   
    
fork = BranchPythonOperator(
    task_id='branch',
    python_callable=branch,
    provide_context=True,
    dag=dag)
 
t4 = PythonOperator(
    task_id='model_good_for_production',
    python_callable=model_good_for_production,
    provide_context=True,
    dag=dag
)   

t5 = PythonOperator(
    task_id='model_needs_retrain',
    python_callable=model_needs_retrain,
    provide_context=True,
    dag=dag
)   
    
sensors >> t1 >> t2 >> t3 >> fork >> [t4,t5]
