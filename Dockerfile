FROM python:3.6-slim

VOLUME /roboflow
WORKDIR /roboflow
COPY README.md setup.py /roboflow/

# Add GCC + CV2 dependencies
RUN apt-get update \
    && apt-get -y install python3-dev build-essential libsm6 libxext6

RUN pip3 install -U pip \
    && pip3 install -e ".[dev]"

CMD ["/bin/bash"]
