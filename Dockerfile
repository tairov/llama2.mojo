# https://github.com/modularml/mojo/blob/main/examples/docker/Dockerfile.mojosdk
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# Example command line:
# Use no-cache to force docker to rebuild layers of the image by downloading the SDK from the repos
# docker build --no-cache \
#    --build-arg AUTH_KEY=<your-modular-auth-key>
#    --pull -t modular/mojo-v0.2-`date '+%Y%d%m-%H%M'` \
#    --file Dockerfile.mojosdk .

FROM ubuntu:20.04

ARG DEFAUL_TZ=America/Los_Angeles
ENV DEFAULT_TZ=$DEFAULT_TZ
ARG MODULAR_HOME=/home/user/.modular
ENV MODULAR_HOME=$MODULAR_HOME

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive $DEFAULT_TZ apt-get install -y \
    tzdata \
    vim \
    sudo \
    curl \ 
    python3 \
    pip \
    wget \
    && python3 -m pip install \
    jupyterlab \
    ipykernel \
    matplotlib \
    ipywidgets \
    gradio 

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh > /tmp/miniconda.sh \
    && chmod +x /tmp/miniconda.sh \
    && /tmp/miniconda.sh -b -p /opt/conda

ARG AUTH_KEY=DEFAULT_KEY
ENV AUTH_KEY=$AUTH_KEY

RUN curl https://get.modular.com | MODULAR_AUTH=$AUTH_KEY sh - \
    && modular install mojo 

RUN useradd -m -u 1000 user
RUN chown -R user $MODULAR_HOME

ENV PATH="$PATH:/opt/conda/bin:$MODULAR_HOME/pkg/packages.modular.com_mojo/bin"

RUN conda init 
RUN pip install gradio

USER user
WORKDIR $HOME/app

COPY --chown=user . $HOME/app
RUN wget -c https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
RUN wget -c https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
RUN wget -c https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin

# CMD ["mojo", "llama2.mojo"]
CMD ["python3", "gradio_app.py"]