# Pull python 3 image
FROM python:3
# Create a work dir
WORKDIR /usr/src/app
# copy requirements.txt into workdir created above
COPY requirements.txt ./
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
# Install all requirements
RUN python3 -m pip install --user --no-cache-dir -r requirements.txt
# Copy entire project into workdir
COPY . .
# Run our app with output
CMD ["python","-u","app.py"]