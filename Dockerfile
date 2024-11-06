FROM python:3.12
WORKDIR /usr/local/app

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY src ./src
COPY run.sh ./run.sh
EXPOSE 8000

RUN mkdir logs

# Setup an app user so the container doesn't run as the root user
RUN useradd app
#USER app

CMD ["./run.sh"]