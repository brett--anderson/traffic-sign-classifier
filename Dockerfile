FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

MAINTAINER Craig Citro <craigcitro@google.com>

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python3-dev \
        rsync \
        software-properties-common \
        unzip \
        libgtk2.0-0 \
        git \
		tcl-dev \
		tk-dev \	
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

ADD https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh tmp/Miniconda3-4.2.12-Linux-x86_64.sh
RUN bash tmp/Miniconda3-4.2.12-Linux-x86_64.sh -b
ENV PATH $PATH:/root/miniconda3/bin/

COPY environment-gpu.yml  ./environment.yml
RUN conda env create -f=environment.yml --name carnd-term1 --debug -v -v

# cleanup tarballs and downloaded package files
RUN conda clean -tp -y

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Term 1 workdir
RUN mkdir /src
WORKDIR "/src"

# Make sure CUDNN is detected
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64/:$LD_LIBRARY_PATH
RUN ln -s /usr/local/cuda/lib64/libcudnn.so.5 /usr/local/cuda/lib64/libcudnn.so

# TensorBoard
EXPOSE 6006
# Jupyter
EXPOSE 8888
# Flask Server
EXPOSE 4567



### Setup directories and copy files across

RUN mkdir /opt/traffic_sign_predictor
RUN mkdir /opt/traffic_sign_predictor/data
RUN mkdir /opt/traffic_sign_predictor/traffic-signs-data/

COPY Traffic_Sign_Classifier.ipynb /opt/traffic_sign_predictor/
COPY signnames.csv /opt/traffic_sign_predictor/

RUN apt-get update && apt-get install -y --no-install-recommends \
        unzip \
        wget

RUN wget https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip && \
    unzip traffic-signs-data.zip && cp *.p /opt/traffic_sign_predictor/traffic-signs-data/ && \
    rm *.p && \
    rm *.zip


COPY run.sh /opt/
RUN chmod +x /opt/run.sh
ENTRYPOINT ["/opt/run.sh"]