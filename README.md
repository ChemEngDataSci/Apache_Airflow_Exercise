# Apache_Airflow_Exercise

The purpose of this exercise is to intergrate Apache Airflow to automate a ML pipeline. This is a template that can be used as reference, more features can be added in such as grid search on parameters to refit the model. The default arguments can be adjusted so that it refreshs daily/weekly/ etc. 

Airflow is set up within a docker container (Puckel docker-airflow-master) with a LocalExecutor.

The Airflow pipeline is to simulate when new results are available, it is scored against the current ML model (e.g. gradient boosting) to see if the model performance is still within the threshold. If it is not, then the model is automatically refitted to improve accuracy. 

Features used: PythonOperator, BranchPythonOperator, FileSensor, XCom
