"# youtube_sentimental_analyzer" 



conda create -p venv python==3.11 -y
conda activate D:\bybappy\youtube_sentimental_analyzer\venv


pip install -r requirements.txt


## DVC

dvc init

dvc repro

dvc dag



## AWS

aws configure



### Json data demo in postman

http://localhost:5000/predict

```python
{
    "comments": ["This video is awsome! I loved a lot", "Very bad explanation. poor video"]
}
```



chrome://extensions