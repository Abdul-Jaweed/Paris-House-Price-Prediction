import pymongo
import pandas as pd
import os
from logger import logging
from dotenv import load_dotenv

load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")

DATABASE_NAME = "RealState"
COLLECTION_NAME = "paris"
RAW_DATA_FOLDER = "raw_data"
CSV_FILE_NAME = "raw.csv"

def fetch_data_from_mongodb():
    # Connect to MongoDB using the provided URL
    client = pymongo.MongoClient(mongo_db_url)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    # Fetch data from MongoDB and exclude the "_id" field
    data = list(collection.find({}, {"_id": 0}))

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Store the data as a CSV file in the "raw_data" folder
    if not os.path.exists(RAW_DATA_FOLDER):
        os.makedirs(RAW_DATA_FOLDER)

    file_name = os.path.join(RAW_DATA_FOLDER, CSV_FILE_NAME)
    df.to_csv(file_name, index=False)

    logging.info(f"Fetching and storing data successfully in {file_name}")

    return df
