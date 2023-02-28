local:
	docker run -it --rm --volume "$(PWD)":/home/jovyan -p 8888:8888 \
		-e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
		pangeo/pangeo-notebook:2023.02.08 jupyter lab --ip 0.0.0.0
