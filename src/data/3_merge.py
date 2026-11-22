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

def load_data(flat_path : str , house_path):
    logger.debug('Loading data Function Started...')
    logger.debug('Loading data')

    if not os.path.isfile(flat_path):
        logger.error("Input CSV file does not exist: %s", flat_path)
        raise FileNotFoundError(f"Input CSV file does not exist: {flat_path}")

    if not os.path.isfile(house_path):
            logger.error("Input CSV file does not exist: %s", house_path)
            raise FileNotFoundError(f"Input CSV file does not exist: {house_path}")
    
    try:
        # loaded data
        house = pd.read_csv(house_path , index_col=0)
        logging.debug('House Data Loaded Successfully %s' , house.shape)

        flat = pd.read_csv(flat_path , index_col=0)
        logging.debug('Flat Data Loaded Successfully %s' , flat.shape)

        return flat , house
    except Exception:
        logger.exception('Failed to load Data %s ' , house_path , flat_path)
        raise


def merge_data(houses : pd.DataFrame , flats : pd.DataFrame) -> pd.DataFrame:
    logger.debug('Merging data Function Started...')
    logger.debug('Merging data')

    try:
        df = pd.concat([flats,houses])
        logger.debug('Data Merged with %s' , df.shape)

        return df
    except Exception as e:
        logger.error('Error in merging data %s ' , e)
        raise

def save_data(df :pd.DataFrame ,ouput_path :str,output_dir:str)->None:
    logger.debug('save_data function started')
    try:

        os.makedirs(output_dir , exist_ok=True)
        logger.debug('output directory made suceessfully %s' , output_dir)

        df.to_csv(ouput_path)
        logger.debug('Data saved successfully at %s :' , ouput_path)

    except Exception as e:
        logger.exception("Couldn't saved dataframe")
        logger.error('Error occured as %s ' , e)
        raise


def main(flat_path :str , house_path:str , ouput_path :str,output_dir:str):
    try:
        logger.info('main function execution started')
        flat , house = load_data(flat_path=flat_path , house_path=house_path)
        logger.debug('Raw Dataframe State %s' , flat.shape , house.shape)

        df = merge_data(flats=flat , houses=house)
        logger.debug('Merged Dataframe State %s' , df.shape)
    
        save_data(df=df , ouput_path=ouput_path , output_dir=output_dir)
    except Exception as e:
        logger.exception('Pipeline Failed')
        logger.error('Error occured as e %s '  ,e)
        raise


if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    logger.info('Current Directory set up %s' , curr_dir)
    logger.info('Home Directory set up %s' , home_dir)

    flat_path = home_dir.as_posix() + '/data/processed/flats/1_flats_cleaned.csv'
    house_path = home_dir.as_posix() + '/data/processed/houses/1_houses_cleaned.csv'

    output_dir = home_dir.as_posix() + '/data/processed/merged'
    logger.debug('output directory set up successfully %s' , output_dir)

    output_path = home_dir.as_posix() + '/data/processed/merged/merged.csv'
    logger.debug('Ouput file path set up successfully %s' , output_path)


    main(
        flat_path=flat_path,
        house_path=house_path,
        ouput_path=output_path,
        output_dir=output_dir
    )
    logger.info('Main Function Ran Successfully')