# Airflow instance

## Run Airflow

Open terminal and select project folder
```shell
cd .../ucu-mlops/final-pipe/airflow
```

```shell
export AIRFLOW_HOME="$(pwd)"
docker compose up --force-recreate
```