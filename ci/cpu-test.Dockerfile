FROM python:3.8.18-slim-bookworm

# Install dependencies
RUN apt-get update \
   && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 gcc g++ \
   && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

COPY env/ env/

RUN pip3 install -r env/requirements-frozen.txt --no-cache-dir --extra-index-url=https://download.pytorch.org/whl/cpu