VERSION=$(shell python3 -c "import spectralDNS; print(spectralDNS.__version__)")

default:
	python setup.py build_ext -i

pip:
	rm -f dist/*
	python setup.py sdist
	twine upload dist/*

tag:
	git tag $(VERSION)
	git push --tags

publish: tag pip

clean:
	git clean spectralDNS -fx
	git clean tests -fx
	git clean demo -fx
	@rm -rf *.egg-info/ build/ dist/ .eggs/