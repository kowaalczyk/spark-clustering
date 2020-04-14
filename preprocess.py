import os
import logging
import sys
import time
import functools
import itertools

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType, FloatType

from pyspark.ml.feature import NGram, CountVectorizer, MinMaxScaler
from pyspark.ml.linalg import Vectors, VectorUDT


class YarnLogger:
    """
    Python logger compatible with yarn logs. Logs can be accessed using:
    ::
        yarn logs -applicationId <your-application-id> | grep '\[PYTHON\]'
    
    Original source:
    https://stackoverflow.com/questions/40806225/pyspark-logging-from-the-executor/40839220
    """

    @staticmethod
    def setup_logger():
        if not "LOG_DIRS" in os.environ:
            sys.stderr.write(
                "Missing LOG_DIRS environment variable, pyspark logging disabled"
            )
            return

        file = os.environ["LOG_DIRS"].split(",")[0] + "/pyspark.log"
        logging.basicConfig(
            filename=file,
            level=logging.INFO,
            format="[PYTHON] %(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        )

    def __getattr__(self, key):
        return getattr(logging, key)


def timed(func):
    """ Logs execution time of the function """
    logger = YarnLogger()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        logger.info(f"{func.__name__} finished in {duration:.2f}s")
        return ret

    return wrapper


def logged_kwargs(func):
    """ Logs kwargs passed to the function """
    logger = YarnLogger()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"{func.__name__} keyword arguments:")
        for arg_name, arg_value in kwargs.items():
            logger.info(f"{arg_name}: {arg_value}")
        ret = func(*args, **kwargs)
        return ret

    return wrapper


@timed
def read_and_repartition(in_file, spark: SparkSession):
    logger = YarnLogger()

    n_executors = int(spark.conf.get("spark.executor.instances", "1"))
    n_cores = int(spark.conf.get("spark.executor.cores", "1"))
    n_partitions = 2 * n_executors * n_cores
    logger.info(f"reading {in_file} into {n_partitions} partitions")

    df = (
        spark.read.csv(in_file, sep="\t", header=True, inferSchema=True)
        .repartition(n_partitions)
        .cache()
    )
    logger.debug(f"df.schema={df.schema}")
    return df


@udf(returnType=ArrayType(StringType()))
def toTokenList(x):
    """ String to list of characters (tokens) """
    return list(x)


@udf(returnType=VectorUDT())
def toDenseVector(x):
    """ Sparse vector to dense vector """
    return Vectors.dense(x)


@udf(returnType=FloatType())
def vectorSum(x):
    return float(x.norm(0))


@timed
@logged_kwargs
def preprocess_df(
    in_file, n_shingles: int, use_binary_vectors: bool, use_dense_vectors: bool = False
):
    """
    Loads TSV from in_file, performs preprocessing and saves to out_file.
    WARNING: dense vectors can extend computation time by more than 70x
    """
    spark = SparkSession.builder.getOrCreate()

    logger = YarnLogger()
    logger.info("preprocessing started")
    logger.info(f"in_file = '{in_file}'")
    logger.info(f"n_shingles = ''")

    raw_df = read_and_repartition(in_file, spark)
    logger.debug(f"raw_df.schema={raw_df.schema}")

    token_column_name = "sequence_tokens"
    df_tokens = raw_df.select(
        raw_df["Entry"].alias("entry"),
        raw_df["Entry name"].alias("entry_name"),
        toTokenList("Sequence").alias(token_column_name),
    )
    logger.debug(f"df_tokens.schema={df_tokens.schema}")

    ngram = NGram(n=n_shingles, inputCol=token_column_name, outputCol="sequence_ngrams")
    df_shingle = ngram.transform(df_tokens)
    logger.debug(f"df_shingle.schema={df_shingle.schema}")

    vectorizer = CountVectorizer(
        binary=use_binary_vectors,
        inputCol=ngram.getOutputCol(),
        outputCol="sequence_vector",
    )
    vectorizer = vectorizer.fit(df_shingle)
    df_vectorized = vectorizer.transform(df_shingle).cache()
    logger.debug(f"df_vectorized.schema={df_vectorized.schema}")

    samples_before_dropped = df_vectorized.count()
    df_dropped = df_vectorized.where(vectorSum(vectorizer.getOutputCol()) > 0).cache()
    samples_dropped = samples_before_dropped - df_dropped.count()
    logger.warning(
        f"dropped {samples_dropped} samples with feature vector norm == 0 (out of {samples_before_dropped} total samples)"
    )

    feature_column_name = "features"
    if use_dense_vectors:
        df_features = df_dropped.select(
            df_dropped["entry"],
            df_dropped["entry_name"],
            toDenseVector(vectorizer.getOutputCol()).alias(feature_column_name),
        )
    else:
        # vectorizer output is sparse by default
        df_features = df_dropped.select(
            df_dropped["entry"],
            df_dropped["entry_name"],
            df_dropped[vectorizer.getOutputCol()].alias(feature_column_name),
        )
    logger.debug(f"df_features.schema={df_features.schema}")

    vector_density = "dense" if use_dense_vectors else "sparse"
    vector_type = "binary" if use_binary_vectors else "count"
    out_file = (
        f"/data/df_{n_shingles}-shingles_{vector_density}-{vector_type}-vectors.parquet"
    )
    df_features.write.parquet(out_file, mode="overwrite")
    logger.info(f"out_file={out_file}")


if __name__ == "__main__":
    spark = SparkSession.builder.appName("Preprocessing").getOrCreate()

    YarnLogger.setup_logger()
    logger = YarnLogger()

    data_file = sys.argv[1]  # TODO: ArgumentParser

    n_shingles_vals = range(3, 6)
    use_binary_vectors_vals = [True, False]
    use_dense_vectors_vals = [False]
    # not using dense vectors after initial experiment - see the README

    grid = list(
        itertools.product(
            n_shingles_vals, use_binary_vectors_vals, use_dense_vectors_vals
        )
    )
    n_iter = len(grid)
    start = time.time()
    for idx, (n_shingles, use_binary_vectors, use_dense_vectors) in enumerate(grid):
        preprocess_df(
            data_file,
            n_shingles=n_shingles,
            use_binary_vectors=use_binary_vectors,
            use_dense_vectors=use_dense_vectors,
        )

        elapsed_time = time.time() - start
        n_completed_iters = idx + 1
        time_per_iter = elapsed_time / n_completed_iters
        logger.info(
            f"completed {n_completed_iters} in {elapsed_time:.2f}s ({time_per_iter:.2f}s/it)"
        )

        if n_completed_iters < n_iter:
            remaining_iters = n_iter - n_completed_iters
            est_time = time_per_iter * remaining_iters
            logger.info(f"estimated remaining time: {est_time:.2f}s")

    total_time = time.time() - start
    logger.info(
        f"preprocessing finished in {total_time:.2f}s ({total_time/60:.2f} minutes)"
    )
    spark.stop()
