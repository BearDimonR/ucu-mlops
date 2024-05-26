# Airflow instance

## Local Dev Tutorial

Open terminal and select project folder
```shell
cd .../ucu-mlops/airflow
```

```shell
export AIRFLOW_HOME="$(pwd)"
docker compose up --force-recreate
```

To run airflow tasks locally for debug purposes use PYTHONPATH (in PyCharm they are set up automatically)
```shell
PYTHONPATH="$(pwd)" airflow dags list
```