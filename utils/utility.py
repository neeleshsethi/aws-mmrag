import time
import boto3
import random
import os

suffix = random.randrange(200, 900)
boto3_session = boto3.session.Session()
region_name = boto3_session.region_name
iam_client = boto3_session.client('iam')
account_number = boto3.client('sts').get_caller_identity().get('Account')
identity = boto3.client('sts').get_caller_identity()['Arn']

# S3
sts_client = boto3.client('sts')
s3_client = boto3.client('s3')
config_filename = ''

def read_key_value(file_path, key1):
    with open(file_path, 'r') as file:
        for line in file:
            key_value_pairs = line.strip().split(':')
    kb_id = read_key_value(config_filename, 'KB_id')
    ds_id = read_key_value(config_filename, 'DS_id')
    region_name = read_key_value(config_filename, 'Region')
    bucket_name = read_key_value(config_filename, 'S3_bucket_name')
    kb_id = read_key_value(config_filename, 'KB_id')
    kb_id = read_key_value(config_filename, 'KB_id')
    kb_id = read_key_value(config_filename, 'KB_id')

def interactive_sleep(seconds: int):
    dots = ''
    for i in range(seconds):
        dots += '.'
        print(dots, end='\r')
        time.sleep(1)
    print('Done!')


#Upload to S3
def uploadDirectory(path,bucket_name):
    for root,dirs,files in os.walk(path):
        for file in files:
            s3_client.upload_file(os.path.join(root,file),bucket_name,file)

def empty_versioned_s3_bucket(bucket_name):
    s3r = boto3.resource('s3')
    bucket = s3r.Bucket(bucket_name)
    bucket.object_versions.delete()
    return True

    