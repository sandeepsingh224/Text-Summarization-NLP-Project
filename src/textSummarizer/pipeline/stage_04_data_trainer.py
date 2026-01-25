from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.model_tranier import ModelTrainer


class DataModelTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer= ModelTrainer(config=model_trainer_config)
        model_trainer.train()