"# youtube_sentimental_analyzer" 
flow to write code 
use dvc 
data_ingestion ->data_preprocessing->model_building
use mlflow
model_evaluation


mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root s3://youtube-sentimental-bucket

######
conda create -p venv python==3.11 -y
conda activate D:\bybappy\youtube_sentimental_analyzer\venv


pip install -r requirements.txt


## DVC

dvc init

dvc repro

dvc dag



## AWS

1)aws console login 
2)create iam user give adminstrative policy 
3)create s3 bucket 
4)create ec2 instance 
use pem method as we want to directly use the interface 
click on connect it will launch ec2 interface 
use this command:
sudo apt update
sudo apt install python3-pip
sudo apt install pipenv
sudo apt install virtualenv
mkdir mlflow
cd mlfow
pipenv install mlflow
pipenv install awscli
pipenv install boto3
pipenv shell #to run virtualenv


5)aws configure 
use:aws configure list #to check the credentials 
or 
use:aws s3 ls  #to check list of s3 bucket
or 
usse:aws sts get-caller-identity #to check iam user id


6)start mlflow server write the s3 bucket name you want to start 
mlflow server -h 0.0.0.0 --default-artifact-root s3://bucketname
example:mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root s3://youtube-sentimental-bucket

note : it start will start port in 5000 so we have to map the port of ec2
a)select ec2 in which mlfow is running 
b)go to security -> security groups ->edit inbound rules ->add rules 
c)5000 port -> click on serch icon tab select anywhere option
d)save rules
e)go back selct instance selct ipv4 dns this will open mlflow server 
ip4dnslink:5000
or 
http://ip4dnslink:5000 #without remove 's' from https 
#check .env for example


if you eant to stop virtual machine in ec2 
use:deactivate

after that stop ec2 instance

if you want to restart 
1)retart the ec2 instance
2)
click connect ec2 interface will open 
(if you want to do it using 3rd party like mobxtream use ssh method command 
ssh -i "path of pem or putty file " machinenameu@public ip or dns 
) 

3)cd mlflow

4)pipenv shell

5)mlflow server --default-artifact-root s3://s3bucketname -h 0.0.0.0 -p 5000  


6) as we have not use elastic ip to ec2 public ipv4 and dns will change automatically 

7) copy new dns or ipv4 in our .env and click ctrl+s



### Json data demo in postman

http://localhost:5000/predict

```python
{
    "comments": ["This video is awsome! I loved a lot", "Very bad explanation. poor video"]
}
```



chrome://extensions

######
flow to write code 
use dvc 
data_ingestion ->data_preprocessing->model_building
use mlflow
model_evaluation

#####
some error related 
## ⚠️ Common Error: MLflow `NoSuchBucket` During Artifact Upload

### 🧾 Problem Statement
When running `dvc repro` or executing an MLflow-tracked script, the following error may occur:



### 💡 Cause
- This error occurs if the **MLflow experiment** was created with an **incorrect or non-existent S3 bucket** in its `artifact_location`.
- MLflow **does not update** the artifact location even if the server is restarted with a new `--default-artifact-root`.

---

### ✅ Solution

1. **Create a new experiment with the correct S3 bucket**:
    ```python
    mlflow.set_experiment("your-new-experiment-name")
    ```

2. **List all experiments to verify their artifact locations**:
    ```python
    from mlflow.tracking import MlflowClient
    import os
    from dotenv import load_dotenv

    load_dotenv(override=True)
    client = MlflowClient(tracking_uri=os.getenv("MLFLOW_SERVER_TRACKING_URI_EC2"))

    for exp in client.search_experiments():
        print(f"Name: {exp.name}")
        print(f"ID: {exp.experiment_id}")
        print(f"Artifact Location: {exp.artifact_location}")
    ```

3. **(Optional)** Delete the experiment pointing to the wrong S3 bucket:
    > ⚠️ Warning: This permanently deletes all runs associated with the experiment.
    ```python
    client.delete_experiment("experiment_id_here")
    ```

---

### 🛡️ Best Practice
- Always double-check the `--default-artifact-root` before creating an experiment.
- If you ever switch S3 buckets, create a **new experiment** — MLflow does not update artifact locations for existing experiments.




