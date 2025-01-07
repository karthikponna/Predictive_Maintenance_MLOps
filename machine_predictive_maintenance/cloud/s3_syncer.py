
import os


class S3Sync:

    """
    A class to handle synchronization of folders to and from an S3 bucket.
    """

    def sync_folder_to_s3(self,folder,aws_bucket_url):

        """
        Synchronizes a local folder to an S3 bucket.

        Args:
            folder (str): The path to the local folder to be synced.
            aws_bucket_url (str): The S3 bucket URL where the folder will be synced.
        """

        command = f"aws s3 sync {folder} {aws_bucket_url} "
        os.system(command)

    def sync_folder_from_s3(self,folder,aws_bucket_url):

        """
        Synchronizes an S3 bucket to a local folder.

        Args:
            folder (str): The path to the local folder where the S3 data will be synced.
            aws_bucket_url (str): The S3 bucket URL from where the folder will be synced.
        """
        
        command = f"aws s3 sync  {aws_bucket_url} {folder} "
        os.system(command)
