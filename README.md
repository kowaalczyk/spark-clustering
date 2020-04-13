#  Spark clustering

This repo contains:
- hadoop & spark setup using ansible
- comparison of clustering approaches on human protein dataset


## Project setup

In order to run the code on a pyspark running on yarn, you need to:
1. Download and install ansible ([instructions](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html))
2. Set up a cluster of at least 3 computers (you can use [`cluster.sh`](cluster.sh))
   to do this automatically on DigitalOcean - see notes in the section below)
3. Specify IP addresses of your machines in [`deploy/hosts` file](deploy/hosts)
4. Use GNU Make to set up the cluster and run all experiments:
```bash
make  # runs all steps in sequence
```

Individual steps can be also performed separately:
```bash
make deploy  # install necessary software on the cluster computers
make notebook  # start a service with jupyter notebook on the master node
make preprocess  # run preprocessing script (preparing data for experiments)
make clustering  # run clustering script (experiments)
```

If you are using Windows, instead of make you can run respective commands
manually (just copy them from the Makefile to a terminal - both cmd and ps should work).


### Running on DigitalOcean

I developed and performed all experiments running on a cluster of machines hosted
on DigitalOcean, they were the cheapest cloud for this purpose and have a much
simpler interface than AWS, Azure or Google Cloud.

CPU-optimized droplets (equivalent of instances) specs and prices at the moment of writing this were:
```
Slug               Memory    VCPUs    Disk    Price Monthly    Price Hourly
c-2                4096      2        25      40.00            0.060000
c-4                8192      4        50      80.00            0.119000
c-8                16384     8        100     160.00           0.238000
c-16               32768     16       200     320.00           0.476000
```

During experiments, I used c-2 instance as master node and 2 x c-4 or 4 x c-4 as slave nodes.
Number of slaves and selection of nodes can be customized in cluster.sh and 

> TODO: Add DigitalOcean promo code and cli installation instructions


## Experiments - clustering human proteins

> TODO: Describe digitalocean setup (2 slaves, each with 8GB RAM, 4CPU, 50GB SSD)
> TODO: Describe how to change digitalocean setup (cluster.sh, deploy/variables.yml)
> TODO: Describe how to fill-out hosts file


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
