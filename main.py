from src.textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.textSummarizer.pipeline.stage_04_data_trainer import DataModelTrainingPipeline
from src.textSummarizer.pipeline.stage_05_data_evaluation import DataModelEvaluationPipeline
from src.textSummarizer.logger import logging



# STAGE_NAME = "Data Ingestion stage"
# try:
#    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_ingestion = DataIngestionTrainingPipeline()
#    data_ingestion.main()
#    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logging.exception(e)
#         raise e



# STAGE_NAME = "Data Validation stage"
# try:
#    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_validation = DataValidationTrainingPipeline()
#    data_validation.main()
#    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logging.exception(e)
#         raise e



# STAGE_NAME = "Data transformation stage"
# try:
#    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    data_transformation = DataTransformationTrainingPipeline()
#    data_transformation.main()
#    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logging.exception(e)
#         raise e



# STAGE_NAME = "Model Trainer stage"
# try:
#    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    model_trainer = DataModelTrainingPipeline()
#    model_trainer.main()
#    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logging.exception(e)
#         raise e



STAGE_NAME = "Model Evaluation stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_evaluation = DataModelEvaluationPipeline()
   model_evaluation.main()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise e