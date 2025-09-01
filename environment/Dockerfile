# hash:sha256:c68da0ebc9707e424541724d674a672aca43392bad36e205b299b3c6a4fa629e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    allensdk==2.16.2 \
    sklearn==0.0
