local:
	docker run -it --rm --volume "$(PWD)":"$(HOME)" -p 8888:8888 \
		pangeo/pangeo-notebook:2021.10.19 jupyter lab --ip 0.0.0.0 "$(HOME)"
