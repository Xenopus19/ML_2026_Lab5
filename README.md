# Lab 5
There are both parts of the lab in this repo.

### Part 1

The service part is placed in lab_5_1. Here, app.py is the web server code itself, train.py is code used to train a model, and test.py is a test script which can be used to quickly send some test requests to the server. Dockerfile used for the solution's building is named api.Dockerfile. 

To build and test the API with Docker, open a terminal in `lab_5_1` and build the image:

```sh
docker build -f api.Dockerfile -t iris-api .
```

Start the API container and publish its port:

```sh
docker run --rm --name iris-api -p 8000:8000 iris-api
```

Keep that terminal running. In a second terminal, also in `lab_5_1`, run the test script:

```sh
python test.py
```

The script sends an Iris sample to `http://127.0.0.1:8000`, polls for the result, and prints the prediction. Stop the container with `Ctrl+C` when finished.