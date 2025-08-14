FROM python:3.12-slim-bookworm

# Set Workspace directory
ARG WORKSPACE=/workspaces/hypertoolz
ENV WORKSPACE=${WORKSPACE}

# Set an argument for the Poetry version
ARG POETRY_VERSION=1.8.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV_IN_PROJECT=1
ENV POETRY_NO_INTERACTION=1
ENV PATH=$POETRY_HOME/bin:$PATH

# Install Poetry and any build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git \
    && curl -sSL https://install.python-poetry.org | POETRY_VERSION=$POETRY_VERSION python3 - \
    && apt-get autoremove -y curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ${WORKSPACE}/
RUN cd $WORKSPACE && poetry install --no-root && rm -rf /root/.cache/pypoetry/{cache,artifacts}

WORKDIR ${WORKSPACE}