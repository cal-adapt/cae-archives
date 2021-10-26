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

Running on AWS ECS
------------------

You will need to pass your AWS credentials as environment variables to Docker:

    export AWS_ACCESS_KEY_ID=
    export AWS_SECRET_ACCESS_KEY=

Then create a Docker context to read these variables:

    docker context create ecs ecsenv

Now start up JupyterLab in the container:

    docker --context ecsenv compose up

When finished, be sure to tear down the AWS resources to avoid further charges.

    docker --context ecsenv compose down
