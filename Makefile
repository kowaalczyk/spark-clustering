all: deploy preprocess

deploy:
	ANSIBLE_NOCOWS=1 ansible-playbook \
		-i deploy/hosts \
		deploy/deploy-playbook.yml

preprocess:
	ANSIBLE_NOCOWS=1 ansible-playbook \
		-i deploy/hosts \
		--extra-vars "py_file=$@.py" \
		deploy/run-playbook.yml

.PHONY: deploy preprocess
