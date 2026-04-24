from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime
import requests

with DAG(
    dag_id="nasa_apod_postgres",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False
) as dag:

    # 1. Create table
    @task
    def create_table():
        hook = PostgresHook(postgres_conn_id="nasa_apod")
        hook.run("""
        CREATE TABLE IF NOT EXISTS apod_data (
            id SERIAL PRIMARY KEY,
            title TEXT,
            explanation TEXT,
            url TEXT,
            date DATE,
            media_type TEXT
        );
        """)


    @task
    def extract_apod():
        url = "https://api.nasa.gov/planetary/apod"
        params = {
            "api_key": "OdSlC3kbgxUPIfIMPN2oUSq7hkEaB62oWoafAxyF"
        }

        response = requests.get(url, params=params)
        return response.json()


    @task
    def transform_apod_data(response):
        return {
            "title": response.get("title", ""),
            "url": response.get("url", ""),
            "date": response.get("date", ""),
            "media_type": response.get("media_type", ""),
            "explanation": response.get("explanation", "")
        }


    @task
    def load_data_to_postgres(apod_data):
        hook = PostgresHook(postgres_conn_id="nasa_apod")

        hook.run("""
        INSERT INTO apod_data (title, url, date, media_type, explanation)
        VALUES (%s, %s, %s, %s, %s)
        """, parameters=(
            apod_data["title"],
            apod_data["url"],
            apod_data["date"],
            apod_data["media_type"],
            apod_data["explanation"],
        ))


    table = create_table()
    raw = extract_apod()
    cleaned = transform_apod_data(raw)
    load_data_to_postgres(cleaned)