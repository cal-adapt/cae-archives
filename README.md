Caladapt Analytics Engine Notebooks
===================================

Installation
------------

Follow the [Docker installation instructions](https://docs.docker.com/get-docker/)
for your platform.

Getting started
---------------

There is a Makefile for conveniently running JupyterLab off of the
pangeo-notebook image. To run on your local machine:

    make local

Or, to run on the Elastic Container Service (ECS) on AWS:

    make ecs
