Caladapt Analytics Engine Notebooks
===================================

Installation
------------

Follow the [Docker installation instructions](https://docs.docker.com/get-docker/)
for your platform.

Accessing AWS resources
-----------------------

You will need to pass your AWS credentials as environment variables to Docker:

    export AWS_ACCESS_KEY_ID=
    export AWS_SECRET_ACCESS_KEY=

If you don't have a default AWS region set you will need to export that too:

    export AWS_DEFAULT_REGION=us-west-1

Getting started
---------------

There is a Makefile for conveniently running JupyterLab off of the
pangeo-notebook image. To run on your local machine:

    make local
