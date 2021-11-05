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

If you don't have a default AWS region set you will need to export that too:

    export AWS_DEFAULT_REGION=us-west-1

Then create a Docker context to read these variables:

    docker context create ecs ecsenv

Now start up JupyterLab in the container:

    docker --context ecsenv compose up

Navigate to the [AWS Load Balancers](https://us-west-1.console.aws.amazon.com/ec2/v2/home?region=us-west-1#LoadBalancers:sort=loadBalancerName), select it and copy the URL next to "DNS name".

![Load balancer](https://user-images.githubusercontent.com/2359002/140552327-3811492d-2ced-4168-9ef3-906cc9e2f618.png)

Append port `:8787` to the URL, open it and you should see the JupyterLab start
page. Copy and paste the Jupyter token from `compose.yaml` into the form and
JupyterLab will start up.

When finished, be sure to tear down the AWS resources to avoid further charges.

    docker --context ecsenv compose down
