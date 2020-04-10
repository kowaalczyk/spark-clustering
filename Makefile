deploy:
	ANSIBLE_NOCOWS=1 ansible-playbook -i deploy/hosts deploy/playbook.yml

.PHONY: deploy
