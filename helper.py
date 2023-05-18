# helper.py
import os, uuid, sys
from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient
from azure.core._match_conditions import MatchConditions
from azure.storage.filedatalake._models import ContentSettings
from fastapi import File, UploadFile
import azure.storage.blob as blob
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

load_dotenv()

class AzureDatalakeStorage:
    def __init__(self, containername: str = "data"):
        self.containername = containername

    def _initialize_storage_account(self):
        storage_account_name = os.getenv('STORAGE_ACCOUNT_NAME')
        storage_account_key =  os.getenv('STORAGE_ACCOUNT_KEY')
        try:  
            global service_client

            service_client = DataLakeServiceClient(
                account_url="{}://{}.dfs.core.windows.net".format(
                    "https", storage_account_name), 
                credential=storage_account_key)
        
        except Exception as e:
            print(e)

    async def upload_file_to_directory(self, dirname: str, filename: str, local_file: UploadFile = File(...)):
        # This function uploads a file to a directory.

        try:
            self._initialize_storage_account()
            # Get a file system client for the "testcontainer" file system.
            file_system_client = service_client.get_file_system_client(file_system=self.containername)

            # Get a directory client for the specified directory.
            directory_client = file_system_client.get_directory_client(dirname) 

            # file_extension = os.path.splitext(filename)[1]
            # u_name = str(uuid.uuid4())
            u_filename = f'{filename}'
            
            # Create a file in the specified directory.
            file_client = directory_client.create_file(u_filename)

            # Get the contents of the local file.
            file_contents = await local_file.read()

            # Append the contents of the local file to the new file.
            file_client.append_data(data=file_contents, offset=0)

            # Flush the data to the new file.
            file_client.flush_data(len(file_contents))

        except Exception as e:
            # Print the exception message.
            print(e)


class AzureBlobStorage:
    def __init__(self, containername: str = "testcontainer"):
        self.containername = containername

    async def upload_file_to_directory(self, filename: str, local_file: UploadFile = File(...)):   
        try:
            connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

            # Create the BlobServiceClient object
            blob_service_client = BlobServiceClient.from_connection_string(connect_str)

            # Create a unique name for the container
            # 파일의 확장자를 가져옵니다.
            file_extension = os.path.splitext(filename)[1]
            u_name = str(uuid.uuid4())
            u_filename = f'{u_name}{file_extension}'
            
            # Create a blob client using the local file name as the name for the blob
            blob_client = blob_service_client.get_blob_client(container=self.containername, blob=u_filename)

            print("Uploading to Azure Storage as blob:\n\t" + filename)

            # Upload the created file
            blob_client.upload_blob(await local_file.read())

            return u_filename

        except Exception as e:
            # Print the exception message.
           print(e)

class LocalStorage:
    def __init__(self, dirname: str = "mydata"):
        self.dirname = dirname

    async def save_file_to_local(self,filename:str,file: UploadFile = File(...)):
            file_path = f"{self.dirname}/{filename}"
            for filename in os.listdir({self.dirname}):
                    os.remove(os.path.join({self.dirname}, filename))
            with open(file_path, "wb") as f:
                    f.write(await file.read())

