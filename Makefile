all: deploy preprocess

deploy notebook:
	ANSIBLE_NOCOWS=1 ansible-playbook \
		-i deploy/hosts \
		deploy/$@-playbook.yml

preprocess clustering:
	ANSIBLE_NOCOWS=1 ansible-playbook \
		-i deploy/hosts \
		--extra-vars "py_file=$@.py" \
		deploy/run-playbook.yml

.PHONY: deploy notebook preprocess clustering
