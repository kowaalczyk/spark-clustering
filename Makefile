deploy/all/hadoop-2.8.5.tgz:
	wget -O $@ http://ftp.man.poznan.pl/apache/hadoop/common/hadoop-2.8.5/hadoop-2.8.5.tar.gz

deploy/master/spark-2.4.5-bin-without-hadoop.tgz:
	wget -O $@ http://ftp.man.poznan.pl/apache/spark/spark-2.4.5/spark-2.4.5-bin-without-hadoop.tgz

deploy: deploy/all/hadoop-2.8.5.tgz deploy/master/spark-2.4.5-bin-without-hadoop.tgz
	ANSIBLE_NOCOWS=1 ansible-playbook -i deploy/hosts deploy/playbook.yml

.PHONY: deploy