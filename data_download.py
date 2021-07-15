# download datasets by using function in util
from util import *

road_safety_accidents_data_url = "http://data.dft.gov.uk.s3.amazonaws.com/road-accidents-safety-data/dftRoadSafetyData_Accidents_2017.zip"


if __name__ == "__main__" :

    download_and_unzip(url=road_safety_accidents_data_url, extract_to='../inputs/')