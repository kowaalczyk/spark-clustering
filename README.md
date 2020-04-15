#  Spark clustering

> Krzysztof Kowalczyk, kk385830

This project contains:
- hadoop & spark setup using ansible
- comparison of clustering approaches on human protein dataset:
   - preprocessing script that can use various feature extraction and
     vectorization settings
   - clustering script that can run and compare clustering algorithms
     and their hyperparameter settings
   - summary notebook that can be used to visualize and compare experiment results
- tools allowing to easily setup a bunch of virtual machines compatible with
  my project on DigitalOcean


## Quick start

To exactly reproduce what I did, follow these steps:

1. Set up an account on DigitalOcean - [this link gives you $100 to spend](https://m.do.co/c/1b35d33cdc21)
2. Install [doctl - the DigitalOcean CLI](https://github.com/digitalocean/doctl#installing-doctl)
3. Set up an API token with **READ & WRITE scope** on [your DigitalOcean dashboard](https://cloud.digitalocean.com/account/api/)
   and paste it into doctl: `doctl auth init`
4. Install [ansible](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html)
5. Configure `cluster.sh` script:
   - create a project via [DigitalOcean web GUI](https://cloud.digitalocean.com/projects/new),
     you can choose whatever name and description you want
   - after creating the project, copy your project id from its URL
     (should be: `https://cloud.digitalocean.com/projects/YOUR-PROJECT-ID-TO-COPY/...`)
     and set variable `DO_PROJECT_ID` at the top of [`cluster.sh`](cluster.sh) to this value
   - optionally, you can also change other settings by changing other variables
     at the top of `cluster.sh` script (number of slaves, os image, etc.)
6. Create the cluster: `./cluster.sh up`, it can take up to a minute
7. Copy IP addresses of master and slave droplets into `deploy/hosts`
8. Install and start hadoop+spark on all machines: `make deploy`
9. Run preprocessing: `make preprocess`
   - optionally, change settings in [`preprocess.py` script](preprocess.py)
     (by default it will create only sparse vectors with 3,4 and 5 shingles
     vectorized in both ways - by count and by presence)
   - this step should take less than **20 minutes** with default settings
     (takes 20-40s per iteration)
10. Run clustering: `make clustering`
   - optionally, change settings in [`clustering.py` script](clustering.py)
     (by default it will run gridsearch on all hyperparameters of KMeans models)
   - this step should take less than **4 hours** with default settings
     (takes 30-60s per iteration)
11. Visualize results in jupyter notebook: `make notebook`
   - to access notebook you need to click the link (with access token) that
     appears at the end of the output
   - once you have jupyter notebook opened in your browser, you can open
     `summary.ipynb` which is automatically uploaded, and run all cells to
     visualize your results
12. To shut down the cluster, run: `./cluster.sh down`
   - if at any point you wish to re-start, you can use `./cluster.sh rebuild`
     to format the droplets to their initial state preserving their IPs
   - there is also `./cluster.sh status` command you can run to check state of
     your running droplets

This setup assumes you will also use DigitalOcean - if you wish to run the project
on other infrastructure, you will have to perform equivalents to steps 1-6 and 12.
For details, see the section at the bottom of this file.


### Accessing cluster resources

All python scripts are submitted to spark cluster running on yarn in cluster mode.

The deployment exposes all useful dashboards publicly (read-only access):
   - spark application manager runs on `MASTER_IP:8088` and allows for tracking
     the progress of submitted jobs, CPU & RAM usage, etc.
   - hadoop file system browser runs on `MASTER_IP:50070`

Deployment automatically sets up `.bash_profile` on the master, so you can run
`ssh MASTER_IP` to connect and run yarn, hdfs or spark commands manually.

This is very useful for accessing runtime logs (log aggregation is enabled, so
they are collected **AFTER** the application finishes running):
`yarn logs -applicationId <your-application-id>`. You can find out what your
application id is from the GUI running on `MASTER_IP:8088`.

Logs from python scripts can be filtered from cluster logs by searching for
**[PYTHON]** keyword, ie: `yarn logs -applicationId <your-application-id> | grep '\[PYTHON\]'`.

Also, to kill a running application, you can use 
`yarn application kill -applicationId <your-application-id>`.

After running `make notebook` the jupyter notebook is running on `MASTER_IP:8888`.


## Clustering experiment - my results

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

Also, since `MinMaxScaler` always returns dense vectors, I decided not to include
it in the preprocessing pipeline, since min-max scaling is only necessary for the
`GaussianMixture` model running on output from non-binary `CountVectorizer`. Without
that change, the preprocessing pipeline would run 30-90 minutes longer.

For the largest shingle size (5), count vectorizer can return empty vectors.
This is due to the fact that it uses `max_vocab_size=2**18` and there are
`20**5 ~= 2**22` possible shingles of length 5. With a dense matrix and standard
64-bit floats, the dataset would occupy 4TB of RAM - with sparse matrices this is
surely much lower, but still not enough to fit on a machine with 16GB RAM. After
further investigation, I found out that the number of samples with empty feature
vectors is exactly 102 - a small portion of the dataset, so I decided to drop them
all (models with cosine similarity cannot handle vectors with norm 0).

Filtered Python logs from the final run of preprocessing are located in 
[`logs/preprocess-sparse-python.log`](logs/preprocess-sparse-python.log).


### Comparison of clustering approaches

#### Gaussian Mixture Model

For the same reason that dense matrices are extremely slow in preprocessing,
GMMs don't work at all - they crash due to low memory during conversion from
sparse to dense matrix in [this file](https://github.com/apache/spark/blob/cee4ecbb16917fa85f02c635925e2687400aa56b/mllib/src/main/scala/org/apache/spark/mllib/stat/distribution/MultivariateGaussian.scala#L121).

I tried to re-partition the data into more partitions before fitting GMM, but
it did not help.

For details, see logs attached in [`logs/clustering-2-all.log`](logs/clustering-2-all.log).

#### K-Means models

The 2 other models ran successfully - I managed to complete the experiment on a
cluster of 2 slaves (both C-8 droplets) in 3.5 hours. The experiment included
testing all possible hyperparameter values (grid search):
- number of shingles in preprocessing
- vectorization in preprocessing (count vs binary)
- number of clusters (k)
- distance metric (cosine vs euclidean)

Document with visualizations and analysis of results is available in 
[`summary/README.md`](summary/README.md). It was generated from a jupyter
notebook you can easily run yourself if you want to re-produce the results.


## Project details

### Deployment and cluster settings

Deployment is configured to work in the following way:
- ansible playbooks (`deploy/*-playbook.yml`) are used to perform actions
  on all of the hosts (master and slaves)
- ansible uses [`deploy/hosts`](deploy/hosts) file to discover the hosts
- common configuration for the playbooks is stored in
  [`deploy/variables.yml`](deploy/variables.yml),
  and can be altered to match your needs
- configuration files, mostly in the form of jinja templates, are stored in
  subfolders of [`deploy`](deploy) folder:
  - [`deploy/all`](deploy/all) contains files copied to all hosts
  - [`deploy/master`](deploy/master) contains files only copied to the master hosts
  these files should not be edited directly, prefer using
  [`deploy/variables.yml`](deploy/variables.yml)
  and variables defined within playbooks to alter cluster parameters
- most common tasks are organized in the [`Makefile`](Makefile)

The most important configuration pieces are variables related to cluster
memory and cpu resources. These settings need to be carefully tuned in order
to make use of all resources we have in the cluster, and proved the most
difficult for me (mostly due to insufficient documentation). I found out the
hard way that while yarn, hdfs and mapreduce configuration is specified wrt
a single cluter node (ie. yarn.resource.memory is actually the amount of 
memory on a given node), while spark submit arguments (ie. number of executors)
are specified for the entire cluster.

Configurations that work for a few slave node sizes that I tested are in the
[`deploy/variables.yml` file](deploy/variables.yml), in `node_config` section.

I'd like to mention 3 resources that proved very helpful when tuning these settings:
- hadoop, yarn and mapreduce settings:
   - https://docs.cloudera.com/HDPDocuments/HDP2/HDP-2.6.4/bk_command-line-installation/content/determine-hdp-memory-config.html
   - https://www.alibabacloud.com/help/doc-detail/28124.htm
- [spark submit cpu & memory settings](https://c2fo.io/c2fo/spark/aws/emr/2016/07/06/apache-spark-config-cheatsheet/)


### Running spark jobs from jupyter notebook

By default, the notebook is **not** configured to utilize all spark resources.
This alloed me to run notebook with results analysis on the master, which for
majority of the time is not occupied, and reserve all computing power of the
slaves for other running experiments.

That said, the notebook uses spark and can be easily configured to run jobs in
the cluster mode using yarn as spark master. To do this, simply execute the
following code at the beginning of your notebook:
```python
sc.stop()

conf = SparkConf().setAppName("notebook").setMaster("yarn")
conf = conf.set("spark.executor.instances", 3)
conf = conf.set("spark.executor.memory", "3g")
conf = conf.set("spark.executor.cores", 3)

sc = SparkContext(conf=conf)
spark = SparkSession(sc)
sc  # new context should point to yarn as the master
```
After executing this code, you have access to `spark` and `sc` variables
which use yarn cluster for running the code, in the same way as the python
scripts do. This is how I developed most of the code, moving it to scripts
only when I made sure it can be executed in the notebook.


### Choice of infrastructure

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

During experiments, I used the following setup:
- c-4 droplet as master node
- 2 x c-8 droplets as slave nodes
- ubuntu 18.04 os image for all nodes

The [invite link with $100 to spend](https://m.do.co/c/1b35d33cdc21)
is more than enough to run this cluster for a few days.

All of the settings described above (and more) can be customized in `cluster.sh`,
simply by changing related variables at the top of the file.

I decided not to use the students server for several reasons:
- it is always overloaded and extremely slow
- no root access, meaning installing software is extremely painful
- no way to restore system to clean settings after I make a mistake
- no way to monitor CPU, DISK or network load caused by my cluster
- long-running jobs (≥1 hour) often get terminated
- project would not be reproducible for anyone outside of the university


### Running on other infrastructure

All steps automated in the Makefile (7-11 from quick start guide):
- copying data and configuration files
- installation and configuration of hadoop, spark and other software
- establishing connevtivity between cluster nodes
- start of hdfs and yarn
- submitting of preprocessing and clustering scripts
- setup of jupyter notebook service

should work on all machines that meet following conditions:
- clean installation of Ubuntu 18.04 OS
  (other ubuntu versions should work too, all I use is `ufw` and `apt`)
- root user access via ssh
- machines can be accessed from each other via the same ip address as the one
  used to access them from wherever you run this project

That said, the only setup I used is the one on DigitalOcean, described above.
Attached `cluster.sh` script creates that configuration by default.

After configuring your machines manually, all you need to do is specify their
ip addresses in `deploy/hosts` file and run `make` commands as instructed in
the quick start guide at the top of this file.

If you are using Windows, instead of make you can run respective commands
manually (just copy them from the Makefile to a terminal - both cmd and ps should work).
