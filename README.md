# mlopsdemo

A mini ML system with MLOPS capabilities (train/deploy/switch/predict) which can 
- train some prediction models for recommendation/classification given user+stores click interaction data set and automatically deploy models to be served by a simple python web API end point.
- train regression models given any regression data such as [Airbnb Pirces](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data?select=AB_NYC_2019.csv)

The task is to predict the next merchant given the click interaction data of users vs stores.
Please see the [data](/data) folder for sample data.




### Steps to run the notebooks/code Used. Run in root mode on Unix Based System

- If `- you@yourmachine:~/somefolder/mlopsdemo/data$` is empty download you can generate
fake data usin the notebook [create_dataset.ipynb](/notebooks/create_dataset.ipynb)

                --you@yourmachine:~/somefolder/mlopsdemo/data$
                    --you@yourmachine:~/somefolder/mlopsdemo/data/clicks.csv
                    --you@yourmachine:~/somefolder/mlopsdemo/data/stores.csv
                    --you@yourmachine:~/somefolder/mlopsdemo/data/users.csv

    ** Note that the datasset in repo is also generate using [faker](https://faker.readthedocs.io/en/master/) and may not actually represent and actual user+store click distribution.

- Using [Docker](https://www.docker.com/) (preferred way to reproduce, requires active internet connection)

    - Install Docker from [here](https://docs.docker.com/get-docker/)
    - Ensure lines related to _keras_ and _tensorflow-gpu_ are `commented` in [setup.py](/mlcore/setup.py) as shown in below snapshot. This is done as Docker image being is of tensorflow base itself. Should be as show below

                .....
                    "node2vec",
                    # "keras",
                    # "tensorflow-gpu",
                ],
    
    - Ensure below lines are `commented` in [svc.py](/svc.py) and in all the notebooks. Should be as show below

            ##UNCOMMENT BELOW 2 LINES IF RUNNING LOCALLY
            #from dotenv import load_dotenv
            #load_dotenv(dotenv_path = '.env')
    
    - Navigate to folder `recsysdemo`. Once there run below command
    
            - you@yourmachine:~/somefolder/mlopsdemo$ chmod a+x rundocker.sh  && ./rundocker.sh 
        
    - Open [http://localhost:9001/tree/notebooks](http://localhost:9001/tree/notebooks) in your browser to examine the notebooks  
    - Open [http://localhost:5000/docs](http://localhost:5000/docs) in your browser to examine the prediction api. 

    -If you system has CUDA + GPU configured properly you can try below command to startup the app.
        
            ~/mlopsdemo$ chmod a+x rundocker.sh  && ./rundocker_gpu.sh 

- Using [Venv](https://docs.python.org/3/library/venv.html) (tested only on python3.8 and ubuntu may require tweaks on your system)

    - Ensure lines related to _keras_ and _tensorflow-gpu_ are `Uncommented` in [setup.py](/mlcore/setup.py) as shown in below snapshot. This is done as Docker image being is of tensorflow base itself. Should be as show below

                .....
                    "node2vec",
                    "keras",
                    "tensorflow-gpu",
                ],
    - Ensure below lines are `Uncommented` in [svc.py](/svc.py) and in all the notebooks. Should be as show below 

                #UNCOMMENT BELOW 2 LINES IF RUNNING LOCALLY
                from dotenv import load_dotenv
                load_dotenv(dotenv_path = '.env')

    - Navigate to folder `recsysdemo`. Once there run below command
        
            - you@yourmachine:~/somefolder/mlopsdemo$ chmod a+x runapplocal.sh  && source ./install_runapplocal.sh 
    
    - Open [http://localhost:8888/tree/](http://localhost:8888/tree/) in your browser. You may have to navigate manually to recsysdemo directory
    - Open [http://localhost:5000/docs](http://localhost:5000/docs) in your browser to examine the prediction api. 




# Summary of Code (Recommendation/Classification)

## Notebooks
- **EDA** : See [eda.ipynb](/notebooks/eda.ipynb) here for insights

- **Modeling/Training/Deployment**  : The [train](/notebooks/train.ipynb) notebook does below things

    - __Loads data/Extracts features/Splits data.__ 

        - All the 3 data points are joined and some feature transformation is done. Features such as _hour_of_day_, _day_of_week_ etc. are extracted

        - Data is split on time. The data befroe _2021-11-07_  is used for training and leftover for evaluating models

    - __Trains Models__ : 3 models and compares them on [Mean Reciprocal Rank(MRR)](https://en.wikipedia.org/wiki/Mean_reciprocal_rank). Train Test Data used is same across all models 
        
       
        - [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html ) based : A classical model which consided all the merchants as classes, the problem was formulated as a class prediction task. Based on features such as   _channel,hour_of_day_, _day_of_week_, _device_ etc. a vector is constructed and then trained on KNN classifier. I was not able to tune params for this. 


        - [Node2Vec](https://snap.stanford.edu/node2vec/) based model which trains a simple model on a graph of user nodes and merchant nodes where the edges betwween them are represented by features such as  _channel,hour_of_day_, _day_of_week_, _device_ etc. 


        - [LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM): Model On sequence of Merchants clicked: This model tries to predict the next merchant based on merchants clicked by  user in past. This info has to be extracted after joining the Click Data and Merchant Info. I was not able to add user context info such as time of day or time difference between sesssions. I convertedd the actual click sequence  to unique click sequence to see if it improves training but didnt atually help much __(m1->m1->m2->m2 is converted to  m1->m2__ . This model has some bugs when running on a machine which does not have GPU/CUDA configured.
    
    
    - __Deploys Model__: The best model which is automatically picked by by web api serving the model

- **Model Switching**  : The [switch_model](/notebooks/switch_model.ipynb) notebook can be used to switch models already trained.

- All above tasks for regression can be achieved using below notebooks
    - [eda_reg](/notebooks/eda_reg.ipynb)
    - [train_reg](/notebooks/train_reg.ipynb)
    - [switch_model_reg](/notebooks/switch_model_reg.ipynb)


## Rest API
### Recommendation/Classification
- **Predict API**  :  [api](/api) is a [fastapi](https://fastapi.tiangolo.com/) based rest_api which exposes the deployed models using a rest endpoint. A simple predict end point shoud come up at [http://localhost:5000/docs#/make%20predictions/predict_predict_post]([http://localhost:5000/docs#/make%20predictions/predict_predict_post  ) whenver the container/app comes up. You can see more info at [http://localhost:5000/docs](http://localhost:5000/docs). 
- **Sample  Requests** : 

    - [cur_req_for_rest.txt](misc/cur_req_for_rest.txt) file containning an exaple api request
    - [req.txt](misc/req.txt) : file containing some json requests

- **Test API Request**

        curl --location --request POST 'http://localhost:5000/predict' \
        --header 'Content-Type: application/json' \
        --data-raw '{
            "req_id":"1242765",
            "id":1000,
            "user_id":64,
            "store_id":99,
            "device":"mobile",
            "platform":"web",
            "channel":"email",
            "created_at":"2021-09-23 00:34:40",
            "num_of_items_req":5
        }'

#### Regression

- **Predict API**  :  [api](/api) is a [fastapi](https://fastapi.tiangolo.com/) based rest_api which exposes the deployed models using a rest endpoint. A simple predict end point shoud come up at [http://localhost:5000/docs#/make%20predictions/predict_predictreg_post](http://localhost:5000/docs#/make%20predictions/predict_predictreg_post) whenver the container/app comes up. You can see more info at [http://localhost:5000/docs](http://localhost:5000/docs). 
- **Sample  Requests** : 

    - [cur_req_for_rest_reg.txt](misc/cur_req_for_rest_reg.txt) file containning an exaple api request
    - [req_reg.txt](misc/req_reg.txt) : file containing some json requests

- **Test API Request**

            curl --location --request POST 'http://localhost:5000/predictreg' \
            --header 'Content-Type: application/json' \
            --data-raw '{
                "id":2539,
                "name":"Clean & quiet apt home by the park",
                "host_id":2787,
                "host_name":"John",
                "neighbourhood_group":"Brooklyn",
                "neighbourhood":"Kensington",
                "latitude":40.64749,
                "longitude":-73.97237,
                "room_type":"Private room",
                "minimum_nights":1,
                "number_of_reviews":9,
                "last_review":"2018-10-19",
                "reviews_per_month":0.21,
                "calculated_host_listings_count":6,
                "availability_365":365
            }
## mlcore package

- [mlcore](/mlcore) is a package of helpers scripts used in notebooks as well as rest_api
- Following scripts deal with training models mainly
    - train_eval_helper_n2v.py
    - train_eval_helper_seq.py
    - train_eval_helper.py

- [predictor.py](/mlcore/mlcore/predictor.py) is what acts as a plug between prediction end point with the trained models and does the model resoultion as well as actual prediction. 
-  Other scripts deal mainly with 
    
    - [data loading](/mlcore/mlcore/data_helper.py) 
    - [model_operations](/mlcore/mlcore/modelops.py)
    - [db_Interactions](/mlcore/mlcore/dbhelper.py)

    



## Other Artefacts

- [models](/models): This folder contains all trained models. You may delete the db and train from scratch.

- [data](/data) :  This folder contains the origial data provied plus some user hisoty data used for training sequence models.
- [mldb.sqlite](data/mldb.sqlite) contains some training reports and latest deployed model info. Serves as a simple model registry in conjunction with the trained models on file

## Future Work:

- Implement batch prediciton endpoint
- More Tests
- absoulute paths at some places should be converted to env vars
- param tuning
