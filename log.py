import os

LOG_FOLDER_PATH = "../logs"

class Run:
    def __init__(self, log_name, config):
        os.makedirs(LOG_FOLDER_PATH, exist_ok=True)

        self.log_name = log_name
        file = open(os.path.join(LOG_FOLDER_PATH, log_name), 'w')
        file.write(f"Logs for {self.get_summary(config)}\n\n")
        file.close

    def log_scalar(self, name, value, epoch=None):
        file = open(os.path.join(LOG_FOLDER_PATH, self.log_name), 'a')
        str_to_write = name + f": {value}\n"
        file.write(str_to_write)
        file.close()
    
    def get_summary(self, config):
        str_to_write = []

        for key in config:
            if key not in {"checkpoints", "dataset_num", "save_log", "log_frequency"}:
                str_to_write.append(f"{key}: {config[key]}")
        
        return "\n".join(str_to_write)