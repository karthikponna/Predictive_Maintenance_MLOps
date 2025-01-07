import sys
from machine_predictive_maintenance.logging import logger


class MachinePredictiveMaintenanceException(Exception):

    """
    Custom exception class for handling errors in the Machine Predictive Maintenance project.
    
    This class provides additional information about the error, including the name of the Python script 
    where the error occurred and the line number. It can be used to log and raise meaningful error messages 
    in the application.

    Args:
        error_message (Exception): The original error message or exception.
        error_details (sys): System module to retrieve exception details like traceback.
    """

    def __init__(self,error_message,error_details:sys):
        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()
        
        self.lineno=exc_tb.tb_lineno
        self.file_name=exc_tb.tb_frame.f_code.co_filename 
    
    def __str__(self):
        return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        self.file_name, self.lineno, str(self.error_message))
        
if __name__=='__main__':
    try:
        logger.logging.info("Enter the try block")
        a=1/0
        print("This will not be printed",a)
    except Exception as e:
           raise MachinePredictiveMaintenanceException(e,sys)