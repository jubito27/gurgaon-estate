import os
import re
import sys
import pathlib
import logging
import pandas as pd
import numpy as np


# ---------------- Logging setup ----------------

def setup_logging(level: int = logging.DEBUG) -> logging.Logger:
    """Configure console logging at the requested level."""
    logger = logging.getLogger("data_preprocessing_houses")
    logger.setLevel(level)

    # Avoid duplicate handlers if this module is imported/run repeatedly
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging(logging.DEBUG)

def _treat_price(x):
    if type(x) == float:
        return x
    else:
        if x[1] == 'Lac':
            return round(float(x[0])/100,2)
        else:
            return round(float(x[0]),2)

def load_data(data_path : str) -> pd.DataFrame:
    logger.debug('Loading data Function Started...')
    logger.debug('Loading data')

    if not os.path.isfile(data_path):
        logger.error("Input CSV file does not exist: %s", data_path)
        raise FileNotFoundError(f"Input CSV file does not exist: {data_path}")

    try:
        # loaded data
        df = pd.read_csv(data_path , index_col=0)
        logging.debug('Data Loaded Successfully %s' , df.shape)

        # droped unuseful columns
        df.drop(columns=['link','property_id'], inplace=True)
        logger.debug('Columns Dropped Successfully name Link and property_id')
        logger.info('Remaining columns %s', df.columns)

        df = df.drop_duplicates()
        logger.debug('Dropped Duplicated Successfully')


        # renamed neccesary columns
        df.rename(columns={'rate':'price_per_sqft'},inplace=True)
        logger.debug('Column renamed Successfully from rate to price_per_sqft')
        logger.info('Current column names : %s', df.columns)

        # fetching actual price values
        df = df[df['price'] != 'Price on Request']
        logger.debug('Fetching actual price values suceessful')

        return df
    except Exception:
        logger.exception('Failed to load Data %s ' , data_path)
        raise


# ------------- Preprocessing Starts------------
def first_preprocessing(df : pd.DataFrame) -> pd.DataFrame:
    logger.debug('first_preprocessing function started...')
    if df.shape[0] == 0:
        logger.error('Empty Dataset received %s' , df.head() , df.shape)

    try:

        # handling society column
        df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()
        df['society'] = df['society'].str.replace('nan','independent')
        logger.debug('Society column handled sucessfully')
    
        # handling price column
        df['price'] = df['price'].str.split(' ').apply(_treat_price)
        logger.debug('price column converted successfully')

        # converting price_per_srft
        df['price_per_sqft'] = df['price_per_sqft'].str.split('/').str.get(0).str.replace('₹','').str.replace(',','').str.strip().astype('float')
        logger.debug('price_per_srft column converted successfully')

        # bedroom column
        df = df[~df['bedRoom'].isnull()]
        logger.debug('Removed Null values from bedroom column')
        df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')
        logger.debug('Bedroom column converted successfully')

        # bathroom column
        df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')
        logger.debug('Bathroom column converted successfully')

        # balcony coumn
        df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No','0')
        logger.debug('Balcony column converted successfully')

        # additionalrooms column
        df['additionalRoom'].fillna('not available',inplace=True)
        logger.debug('Filled Na values in additionroom column')
        df['additionalRoom'] = df['additionalRoom'].str.lower()
        logger.debug('Additionalroom column converted successfully')

        # nooffloor
        df['noOfFloor'] = df['noOfFloor'].str.split(' ').str.get(0)
        df.rename(columns={'noOfFloor':'floorNum'},inplace=True)
        logger.debug('noOfFloor finished successfully and renamed to floorNum')

        # facing column
        df['facing'].fillna('NA',inplace=True)
        logger.info('Null values replaces in facing col with NA')
        logger.debug('Facing column converted successfully')

        # area
        logger.info('Adding New Area column')
        df['area'] = round((df['price']*10000000)/df['price_per_sqft'])
        logger.debug('Area column converted successfully')

        # property_type
        logger.info('Adding New property_type column')
        df.insert(loc=1,column='property_type',value='house')
        logger.debug('property_type column converted successfully')

        logger.info('Processing Level 1 completed for houses dataset')
        logger.info('Current column names ; %s ' , df.columns)
        return df
    except Exception as e:
        logger.exception('Preprocessing Failed')
        logger.error("Some error occured as e %s  " , e)
        raise

def save_data(df :pd.DataFrame ,ouput_path :str,output_dir:str)->None:
    logger.debug('save_data function started')
    try:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            logger.debug('output directory made suceessfully %s' , output_dir)

        df.to_csv(ouput_path)
        logger.debug('Data saved successfully at %s :' , ouput_path)

    except Exception as e:
        logger.exception("Couldn't saved dataframe")
        logger.error('Error occured as %s ' , e)
        raise

def main(data_path : str , output_path : str, output_dir: str)->None:
    try:
        logger.info('main function execution started')
        df = load_data(data_path=data_path)
        logger.debug('Raw Dataframe State %s' , df.head())
        procesed_df = first_preprocessing(df=df)
        logger.debug('Processed Dataframe State %s' , procesed_df.head())
        save_data(df=procesed_df , ouput_path=output_path , output_dir=output_dir)
    except Exception as e:
        logger.exception('Pipeline Failed')
        logger.error('Error occured as e %s '  ,e)
        raise

if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    logger.info('Current Directory set up %s' , curr_dir)
    logger.info('Home Directory set up %s' , home_dir)

    data_path = home_dir.as_posix() + '/data/raw/houses.csv'
    logger.debug('Data Path set up successfully %s' , data_path)

    output_dir = home_dir.as_posix() + '/data/processed/houses'
    logger.debug('output directory set up successfully %s' , output_dir)

    output_path = home_dir.as_posix() + '/data/processed/houses/1_houses_cleaned.csv'
    logger.debug('Ouput file path set up successfully %s' , output_path)


    main(
        data_path=data_path,
        output_path=output_path,
        output_dir=output_dir
    )
    logger.info('Main Function Ran Successfully')

