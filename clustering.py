import os
import logging
import sys
import time
import functools
import itertools

from pyspark.sql import SparkSession

from pyspark.sql import Row, DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator


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

    df = spark.read.parquet(in_file).repartition(n_partitions).cache()
    logger.debug(f"df.schema={df.schema}")
    return df


@timed
def fit(model_algo, df):
    model = model_algo.fit(df)
    return model


@timed
def predict(model, df):
    df_pred = model.transform(df)
    return df_pred


@timed
def evaluate(model_algo, model, df, df_pred, has_distance_cost=True):
    evaluator = ClusteringEvaluator(
        featuresCol=model_algo.getFeaturesCol(),
        predictionCol=model_algo.getPredictionCol(),
        metricName="silhouette",
    )
    silhouette_cost = evaluator.evaluate(df_pred)
    distance_cost = model.computeCost(df) if has_distance_cost else 0.0
    return silhouette_cost, distance_cost


@timed
@logged_kwargs
def perform_clustering_kmeans(in_file, distance, k, ModelCls):
    """
    Performs clustering using ModelCls, which supports both KMeans and Bisecting KMeans.
    """
    spark = SparkSession.builder.getOrCreate()
    logger = YarnLogger()

    df = read_and_repartition(in_file, spark)

    features_column = "features"
    prediction_column = "cluster"
    model_algo = ModelCls(
        k=k,
        distanceMeasure=distance,
        featuresCol=features_column,
        predictionCol=prediction_column,
        seed=42,
    )

    model = fit(model_algo, df)
    df_pred = predict(model, df)

    has_distance_cost = True  # true for all kinds of kmeans models
    silhouette_cost, distance_cost = evaluate(
        model_algo, model, df, df_pred, has_distance_cost
    )

    summary = model.summary
    logger.info(summary)
    result_df = spark.createDataFrame(
        [
            (
                in_file,
                distance,
                k,
                str(ModelCls),
                has_distance_cost,
                silhouette_cost,
                distance_cost,
                min(summary.clusterSizes),
                max(summary.clusterSizes),
                sum(summary.clusterSizes) / len(summary.clusterSizes),
                False,
                0.0,
            )
        ],
        [
            "in_file",
            "distance",
            "k",
            "model",
            "has_distance_cost",
            "silhouette_cost",
            "distance_cost",
            "min_cluster_size",
            "max_cluster_size",
            "mean_cluster_size",
            "has_log_likelihood" "log_likelihood",
        ],
    )
    return result_df


def perform_clustering_gaussian(in_file, k):
    """
    Performs clustering using GMM, returning results_df comparible with the
    one used for KMeans models.
    """
    spark = SparkSession.builder.getOrCreate()
    logger = YarnLogger()

    df = read_and_repartition(in_file, spark)

    # TODO: Perform min-max scaling - but only when CountVectorizer was used in non-binary mode:
    # feature_column_name = "features"
    # scaler = MinMaxScaler(inputCol=vector_column_name, outputCol=feature_column_name)
    # scaler = scaler.fit(df_features)
    # df_scaled = scaler.transform(df_features).cache()

    features_column = "features"
    prediction_column = "cluster"
    model_algo = GaussianMixture(
        k=k, featuresCol=features_column, predictionCol=prediction_column, seed=42
    )

    model = fit(model_algo, df)
    df_pred = predict(model, df)

    has_distance_cost = False
    silhouette_cost, distance_cost = evaluate(
        model_algo, model, df, df_pred, has_distance_cost
    )

    summary = model.summary
    logger.info(summary)
    result_df = spark.createDataFrame(
        [
            (
                in_file,
                "N/A",
                k,
                "GaussianMixture",
                False,
                silhouette_cost,
                distance_cost,
                min(summary.clusterSizes),
                max(summary.clusterSizes),
                sum(summary.clusterSizes) / len(summary.clusterSizes),
                True,
                summary.logLikelihood,
            )
        ],
        [
            "in_file",
            "distance",
            "k",
            "model",
            "has_distance_cost",
            "silhouette_cost",
            "distance_cost",
            "min_cluster_size",
            "max_cluster_size",
            "mean_cluster_size",
            "has_log_likelihood",
            "log_likelihood",
        ],
    )
    return result_df


@timed
@logged_kwargs
def perform_experiment(
    in_files: list, distances: list, ks: list, models: list, result_dfs_list: list
):
    """
    Tests all possible combinations of provided parameters,
    appends results to result_dfs_list.
    """

    grid = list(itertools.product(in_files, distances, ks, models))
    n_iter = len(grid)
    start = time.time()
    for idx, (in_file, distance, k, ModelCls) in enumerate(grid):
        if issubclass(ModelCls, (KMeans, BisectingKMeans)):
            result_df = perform_clustering_kmeans(
                in_file=in_file, distance=distance, k=k, ModelCls=ModelCls
            )
        else:
            result_df = perform_clustering_gaussian(in_file=in_file, k=k)

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

        result_dfs_list.append(result_df)

    total_time = time.time() - start
    logger.info(
        f"clustering experiment finished in {total_time:.2f}s ({total_time/60:.2f} minutes)"
    )


if __name__ == "__main__":
    spark = SparkSession.builder.appName("ClusteringExperiment").getOrCreate()
    YarnLogger.setup_logger()
    logger = YarnLogger()

    result_dfs = []

    in_files = [
        "/data/df_3-shingles_sparse-binary-vectors.parquet",
        "/data/df_3-shingles_sparse-count-vectors.parquet",
        "/data/df_4-shingles_sparse-binary-vectors.parquet",
        "/data/df_4-shingles_sparse-count-vectors.parquet",
        "/data/df_5-shingles_sparse-binary-vectors.parquet",
        "/data/df_5-shingles_sparse-count-vectors.parquet",
    ]
    distances = ["cosine", "euclidean"]
    models = [KMeans, BisectingKMeans]  # GaussianMixture fails - not enough memory
    ks = list(range(2, 20))
    perform_experiment(
        in_files=in_files,
        distances=distances,
        ks=ks,
        models=models,
        result_dfs_list=result_dfs,
    )

    total_results_df: DataFrame = functools.reduce(DataFrame.unionAll, result_dfs)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"/data/results-{timestamp}.parquet"
    total_results_df.write.parquet(filename)
    logger.info(f"saved results: {filename}")

    spark.stop()
