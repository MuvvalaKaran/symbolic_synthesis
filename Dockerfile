from ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive 

RUN apt-get -y update && apt -y install \
    gcc git \
    wget vim curl \
    python3-pip cmake \
    build-essential \
    apt-utils flex bison mona 

# installing spot using apt
RUN wget -q -O - https://www.lrde.epita.fr/repo/debian.gpg | apt-key add -

SHELL ["/bin/bash", "-c"] 

RUN pip3 install pyyaml numpy bidict networkx graphviz ply pybullet pyperplan==1.3 cython IPython svgwrite matplotlib imageio lark-parser==0.9.0 sympy==1.6.1

RUN echo 'deb http://www.lrde.epita.fr/repo/debian/ stable/' >> /etc/apt/sources.list

RUN apt-get -y update && apt -y install spot \
    libspot-dev \
    spot-doc

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# installing system packages
ADD cudd /cudd
ADD PyDD /PyDD

ADD ./ /root/symbolic_planning/src

RUN rm -rf /root/symbolic_planning/src/cudd && rm -rf /root/symbolic_planning/src/PyDD

RUN cd /cudd/ && \
    ./configure --enable-shared \
    --enable-obj --enable-dddmp \
    && make -j4 \
    && make check && make install  

RUN cd /PyDD && python3 setup.py install

RUN cd /root/symbolic_planning/src/src/LTLf2DFA && pip3 install .

RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib' >> ~/.bashrc

WORKDIR /root/symbolic_planning/src

ENTRYPOINT "/bin/bash"