FROM pytorch/pytorch

# if you forked EasyOCR, you can pass in your own GitHub username to use your fork
# i.e. gh_username=myname
ARG gh_username=JaidedAI
ARG service_home="/home/EasyOCR"

# argument for building with poetry
ARG cv2="cv2"
ARG torch="torch"
ARG POETRY_VERSION=1.4.2
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false

ARG DEBIAN_FRONTEND=noninteractive

# Configure apt and install packages
RUN apt-get update -y && \
    apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dev \
    git \
    # cleanup
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists

RUN python -m venv "${POETRY_HOME}" && \
    "${POETRY_HOME}/bin/pip" install -U pip setuptools && \
    "${POETRY_HOME}/bin/pip" install "poetry==${POETRY_VERSION}" && \
    ln -s "${POETRY_HOME}/bin/poetry" "/usr/local/bin"

# Clone EasyOCR repo
RUN mkdir "$service_home" \
    && git clone "https://github.com/$gh_username/EasyOCR.git" "$service_home" \
    && cd "$service_home"

# Build
RUN cd "$service_home" && poetry install --with "$cv2" --with "$torch"
