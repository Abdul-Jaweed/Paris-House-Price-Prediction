from dataset.data_dump import insert_data_to_mongodb
from raw_data.data_fetch import fetch_data_from_mongodb




if __name__ == '__main__':
    try:
        
        # To dump data into mongodb
        #insert_data_to_mongodb()
        
        # Fetch data from mongodb
        fetch_data_from_mongodb()
    
    
    except Exception as e:
        print(e)