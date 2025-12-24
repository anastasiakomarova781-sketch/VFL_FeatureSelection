FROM debian:bookworm as builder

RUN apt update && apt install -y python3 \
                   build-essential \
                   curl \
                   pip \
                   m4

RUN mkdir ~/miniconda3/ ; \
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda3/miniconda.sh ; \
    chmod +x ~/miniconda3/miniconda.sh ; \
    /bin/bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 ; \
    . ~/miniconda3/bin/activate

RUN /root/miniconda3/bin/conda init --all ; \
    /root/miniconda3/bin/conda create -y --name python_3_11_11 python=3.11.11 

SHELL ["/root/miniconda3/bin/conda", "run", "-n", "python_3_11_11", "/bin/bash", "-c"]

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y ; \
    PATH="/root/.cargo/bin:${PATH}"

COPY  "." "/ucbfl"
RUN PATH="/root/.cargo/bin:${PATH}" ; \
    pip install maturin==0.12.20  ; \
    cd /ucbfl ; \
    maturin build --release -m rust/ucbfl_utils/crates/ucbfl_utils/Cargo.toml --out dist 

RUN cd /ucbfl/python/ ; \
    python setup.py bdist_wheel
