# Run a Python application on a Spark standalone cluster
spark-submit --master yarn \
    --deploy-mode cluster \
    --driver-memory $DRIVER_MEMORY \
    --executor-memory $EXECUTOR_MEMORY \
    --executor-cores 1 \
    $SPARK_HOME/examples/src/main/python/pi.py \
    10
