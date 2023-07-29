import pymongo
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")

DATABASE_NAME = "RealState"
COLLECTION_NAME = "paris"
BASE_DIR = "artifact"
FEATURE_STORE_DIR = "data_ingestion/feature_store"
DATASET_DIR = "data_ingestion/dataset"
CSV_FILE_NAME = "raw.csv"
TRAIN_CSV_FILE_NAME = "train.csv"
TEST_CSV_FILE_NAME = "test.csv"

def fetch_data_from_mongodb():
    # Connect to MongoDB using the provided URL
    client = pymongo.MongoClient(mongo_db_url)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    # Fetch data from MongoDB and exclude the "_id" field
    data = list(collection.find({}, {"_id": 0}))

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    return df

def split_train_test_data(df):
    # Split the data into train and test sets (80% train, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, test_df

def data_ingestion():
    # Create the "artifact" directory if it doesn't exist
    os.makedirs(BASE_DIR, exist_ok=True)

    # Fetch data from MongoDB
    data = fetch_data_from_mongodb()

    # Split data into train and test sets
    train_data, test_data = split_train_test_data(data)

    # Store the raw data as a CSV file in the "artifact/data_ingestion/feature_store" directory
    feature_store_path = os.path.join(BASE_DIR, FEATURE_STORE_DIR)
    os.makedirs(feature_store_path, exist_ok=True)
    file_name = os.path.join(feature_store_path, CSV_FILE_NAME)
    data.to_csv(file_name, index=False)

    # Store the train data as a CSV file in the "artifact/data_ingestion/dataset" directory
    dataset_path = os.path.join(BASE_DIR, DATASET_DIR)
    os.makedirs(dataset_path, exist_ok=True)
    train_file_name = os.path.join(dataset_path, TRAIN_CSV_FILE_NAME)
    train_data.to_csv(train_file_name, index=False)

    # Store the test data as a CSV file in the "artifact/data_ingestion/dataset" directory
    test_file_name = os.path.join(dataset_path, TEST_CSV_FILE_NAME)
    test_data.to_csv(test_file_name, index=False)



# if __name__ == '__main__':
#     try:
#         data_ingestion()
#     except Exception as e:
#         print(e)
