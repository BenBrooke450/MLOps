from airflow import DAG
from airflow.providers.http.operators.http import HttpOperator
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
import json

from datetime import datetime

with DAG(
    dag_id="nasa_apod_postgres",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False) as dag:

    @task
    def create_table():
        postgres_hook = PostgresHook(postgres_conn_id="nasa_apod")

        create_table_query = """
            CREATE TABLE IF NOT EXISTS apod_data (
            id SERIAL PRIMARY KEY,
            title VARCHAR(255),
            explanation TEXT,
            url TEXT,
            date DATE,
            media_type varchar(50));"""

        postgres_hook.run(create_table_query)



    ## https://api.nasa.gov/planetary/apod?api_key=OdSlC3kbgxUPIfIMPN2oUSq7hkEaB62oWoafAxyF
    key =  "OdSlC3kbgxUPIfIMPN2oUSq7hkEaB62oWoafAxyF"

    extract_apod = HttpOperator(
        task_id="extract_apod",
        http_conn_id="nasa_api",
        endpoint = "planetary/apod?api_key=OdSlC3kbgxUPIfIMPN2oUSq7hkEaB62oWoafAxyF",
        method='GET',
        response_filter=lambda response:response.json(),

    )


    @task
    def transform_apod_data(response):
        apod_data = {
            "title": response.get("title",""),
            "url": response.get("explanation",""),
            "date": response.get("date",""),
            "media_type": response.get("media_type",""),
            "explanation": response.get("explanation","")
        }

        return apod_data



    @task
    def load_data_to_postgres(apod_data):
        postgres_hook = PostgresHook(postgres_conn_id="nasa_apod")

        insert_query = """
        INSERT INTO apod_data (title, url, date, media_type, explanation)
        VALUES (%s, %s, %s, %s, %s);
            """

        postgres_hook.run(insert_query, parameters = (
                          apod_data["title"],
                          apod_data["explanation"],
                          apod_data["url"],
                          apod_data["date"],
                          apod_data["media_type"]))

        create_tbl = create_table()

        transformed = transform_apod_data(extract_apod.output)

        load_task = load_data_to_postgres(transformed)

        create_tbl >> extract_apod >> transformed >> load_task



