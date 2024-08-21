FROM python:3.10-slim-bookworm

RUN echo "deb https://mirrors.aliyun.com/debian/ bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list
RUN echo "deb https://mirrors.aliyun.com/debian/ bookworm-updates main contrib non-free non-free-firmware" > /etc/apt/sources.list
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libc-dev libffi-dev libgmp-dev libmpfr-dev libmpc-dev \
     libldap2-dev libsasl2-dev
RUN apt-get install -y portaudio19-dev

# install packages
COPY requirements.txt ./

RUN pip install -r requirements.txt --no-cache --index-url=https://pypi.tuna.tsinghua.edu.cn/simple/

EXPOSE 5001

WORKDIR /app/sandbox
# set timezone
ENV TZ=UTC
ENV FLASK_APP=code_runner_app.py

# Copy source code
COPY . /app/sandbox/

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
