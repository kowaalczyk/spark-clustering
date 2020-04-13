#  Spark clustering

This repo contains:
- hadoop & spark setup using ansible
- comparison of clustering approaches on human protein dataset


## Hadoop & spark cluster setup

- use `cluster.sh` to set up / tear down a bunch of droplets on Digital Ocean
- use ansible with configuration files in `ansible/*` folder to install necessary software

```shell
ansible-playbook -i ansible/hosts ansible/hadoop-spark.yml
```


## Experiments - clustering human proteins

- TODO: Describe digitalocean setup (2 slaves, each with 8GB RAM, 4CPU, 50GB SSD)


### Preprocessing

Preprocessing script, implemented in [`preprocess.py`](preprocess.py), creates
parquet files with all combinations of feature generation settings:
- 3,4 or 5 shingle lengths (n-grams)
- binary (present or not) or count values (how many times it appears) for each shingle

The script also allows to use sparse or dense vectors, but it turned out that
using dense vectors takes too much time. By default, `CountVectorizer` 
creates sparse vectors - for a good reason. When trying to convert them to 
dense vectors, the computation takes too long to be feasible (over 1 hour, 
comparing to less than 1 min for sparse vectors). Logs from the single run with
dense vectors enabled are located in [`logs/preprocess-dense-1-iter.log`](logs/preprocess-dense-1-iter.log),
dashboard screenshots in [`logs/preprocessing_dense_vectors` directory](logs/preprocessing_dense_vectors).

What's interesting, is the fact that after writing the dense vectors to disk,
parquet files are of similar size to the sparse ones, which means parquet does
a lot of processing and compression to store files efficiently.

Filtered Python logs from the final run of preprocessing are located in 
[`logs/preprocess-sparse-python.log`](logs/preprocess-sparse-python.log).


### Comparison of clustering approaches

- TODO
