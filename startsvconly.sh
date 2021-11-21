#!/bin/bash
uvicorn svc:app --host 0.0.0.0 --port 5000 & restapi=$!
#jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='' --NotebookApp.password='' & jupnb=$!
#jupyter notebook --allow-root & jupnb=$!
#wait $restapi $jupnb
wait $restapi