deploy:
	ANSIBLE_NOCOWS=1 ansible-playbook -i deploy/hosts deploy/setup-playbook.yml

run:
	ANSIBLE_NOCOWS=1 ansible-playbook -i deploy/hosts deploy/run-playbook.yml

.PHONY: deploy run
