from dataset.data_dump import insert_data_to_mongodb
from raw_data.data_fetch import fetch_data_from_mongodb
from paris.components.data_ingestion import data_ingestion
from paris.components.data_transformation import data_transformation

if __name__ == '__main__':
    try:
        # To dump data into mongodb
        # insert_data_to_mongodb()
        
        # Fetch data from mongodb
        # fetch_data_from_mongodb()
        
        
        data_ingestion()
        data_transformation()
    
    except Exception as e:
        print(e)
