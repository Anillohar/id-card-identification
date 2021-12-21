develop: upgrade-setuptools upgrade-pip requirements-install
	pip install -e .

upgrade-setuptools:
	pip install -U setuptools

upgrade-pip:
	pip install -U pip

requirements-install:
	pip install -r requirements.txt

train_model:
	python model/train_model.py
