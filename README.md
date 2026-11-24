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

### Part 2

The MCP server part is in lab_5_2. plugin.png is a picture of mcp server set up to be accessible via Claude Desktop. Newly added functions are in the server.py file. I've added a add_new_book and link_book_to_author tools.

After trying ui with local ollama, I've changed the app to use Google API instead of Anthropic (because Google gives you a free API key and Anthropic doesn't). As can be seen in queries.png, the app succesfully reaches Google Gemini model, can use the mcp server's tools and, for example, show the database's books.

The app can be run with:

```sh
docker compose up -d 
```