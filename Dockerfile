FROM python:3.10.2

# apt dependencies
RUN apt-get update && apt-get install -y binutils libproj-dev gdal-bin libspatialindex-dev

# Copy requirements.txt
COPY requirements.txt .

RUN pip install -r requirements.txt


# Add code
WORKDIR /opt/Time-Series-Analysis-Tool
COPY . .

EXPOSE 8000