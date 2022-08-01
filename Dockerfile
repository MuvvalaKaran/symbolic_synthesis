# syntax=docker/dockerfile:1

# using ubuntu LTS version
FROM ubuntu:20.04

RUN apt-get -y update && apt -y install make \
    gcc g++ git \
    wget vim curl \
    python3.8 python3.8-dev \
    python 3.8-venv \
    python3-pip python3-wheel \
    cmake \
    build-essential \
    apt-utils


# installing spot using apt
RUN wget -q -O - https://www.lrde.epita.fr/repo/debian.gpg | apt-key add -

RUN apt-get -y update