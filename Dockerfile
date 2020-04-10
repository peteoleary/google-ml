FROM python:3.7.3-stretch

RUN pip install pipenv

RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser

COPY *.py ./
COPY *.zip ./
COPY Pipfile Pipfile.lock ./

RUN pipenv install

RUN wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
RUN unzip cats_and_dogs_filtered.zip

CMD [ "pipenv", "run", "python", "./cats_and_dogs_1_and_2.py" ]
