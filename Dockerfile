FROM rtux/ubuntu-opencv
RUN \
    export DEBIAN_FRONTEND=noninteractive &&\
    apt-get update && \
    apt-mark hold initscripts &&\
    apt-get -y upgrade && \ 
    apt-get install -y python-pip git python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose &&\
    echo --------------------------------dpkg---------------------------- &&\
    dpkg --configure -a &&\
    echo --------------------------------f---------------------------- &&\
    sudo apt-get install -f &&\
    pip install scikit-learn==0.15 && \
    pip install scikit-image==0.11.3 &&\
    mkdir avcs && cd avcs &&\
    git clone https://github.com/InsufficientFund/Car-counter-using-python-opencv.git &&\
    cd Car-counter-using-python-opencv/ &&\
    git checkout develop 

CMD ["python","testClass.py"]
