FROM ubuntu:latest
LABEL authors="laolao"


ENTRYPOINT ["top", "-b"]

FROM python:3.12.4-slim
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./src /code/src

EXPOSE 80
CMD ["uvicorn", "src.group40_leaf.api:app", "--host", "0.0.0.0", "--port", "80"]