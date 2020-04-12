import os
import logging
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType


class YarnLogger:
    # source: https://stackoverflow.com/questions/40806225/pyspark-logging-from-the-executor/40839220
    @staticmethod
    def setup_logger():
        if not 'LOG_DIRS' in os.environ:
            sys.stderr.write('Missing LOG_DIRS environment variable, pyspark logging disabled')
            return 

        file = os.environ['LOG_DIRS'].split(',')[0] + '/pyspark.log'
        logging.basicConfig(filename=file, level=logging.INFO, 
                format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')

    def __getattr__(self, key):
        return getattr(logging, key)


@udf(returnType=ArrayType(StringType()))
def toTokenList(x):
    return list(x)


def fit_pipeline(spark, logger, data_file):
    logger.info("testing if logs work")
    logger.warn("testing if logs work")
    df = spark.read.csv(
        data_file,
        sep="\t",
        header=True,
        inferSchema=True,
    )
    df_tokens = df.select(
        df["Entry"].alias("entry"),
        df["Entry name"].alias("entry_name"),
        toTokenList("Sequence").alias("sequence_tokens"),
    )
    logger.debug(df_tokens.show(3))

if __name__ == "__main__":
    spark = SparkSession.builder.appName("ClusteringExperiment").getOrCreate()
    
    YarnLogger.setup_logger()
    logger = YarnLogger()
    logger.info("My test info statement")
    
    data_file = sys.argv[1]  # TODO: ArgumentParser
    fit_pipeline(spark, logger, data_file)
    
    spark.stop()
