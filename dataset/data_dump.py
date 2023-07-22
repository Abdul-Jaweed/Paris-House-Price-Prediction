import pymongo
import pandas as pd
import json
import os
from logger import logging
from dotenv import load_dotenv

def insert_data_to_mongodb():
    DATA_FILE_PATH = r"dataset\ParisHousing.csv"
    DATABASE_NAME = "RealState"
    COLLECTION_NAME = "paris"
    
    load_dotenv()
    mongo_db_url = os.getenv("MONGO_DB_URL")

    # Connect to MongoDB using the provided URL
    client = pymongo.MongoClient(mongo_db_url)
    
    logging.info("Started Data insert")

    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")
    df.reset_index(drop=True, inplace=True)

    # Convert DataFrame to JSON records and insert into MongoDB
    json_records = json.loads(df.to_json(orient="records"))
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_records)
    
    logging.info("Data inserted successfully")
    
    print("Data inserted successfully.")
