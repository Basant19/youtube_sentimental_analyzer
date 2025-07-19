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
stop ec2 instance

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

7) copy new dns or ipv4 in our .env 


