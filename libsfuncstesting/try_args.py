from transformers import HfArgumentParser
import logging
from src.data_classes import PyTorchTrainingParams



def main():
    parser = HfArgumentParser(PyTorchTrainingParams)
    args = parser.parse_args_into_dataclasses()

    # Create a logger
    logger = logging.getLogger(__name__)

    # Set the log level
    logger.setLevel(logging.DEBUG)

    # Create a log message handler
    handler = logging.StreamHandler()

    # Set the handler level
    handler.setLevel(logging.DEBUG)

    # Create a log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the handler format
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # Log some messages
    logger.info('Start')

    for arg in args:
        logger.debug(str(vars(arg)))

    # Write "Hello World" to a file in the /proj/mounted directory
    with open('/proj/mounted/hello_world.txt', 'w') as f:
        f.write('Hello World\n')

    logger.info('Finish')

if __name__ == "__main__":
    main()