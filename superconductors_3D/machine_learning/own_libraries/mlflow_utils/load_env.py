from os.path import expanduser
import os


def load_env(env_path: str):
    """Loads an .env file in your home directory.
    To access the mlflow api, you need log in credentials. These can be stored in a local environment file which you
    should protect with `chmod 600 .env`.
    In this .env file you should define:
    MLFLOW_TRACKING_USERNAME=your_username
    MLFLOW_TRACKING_PASSWORD=your_password
    MLFLOW_TRACKING_URI=https://ip.on.the.kit.lan

    Returns:
        env: (dict) A dictionary containing the defined key value pairs.
    """
    home = expanduser("~")
    with open(env_path, 'r') as f:
        env = dict()
        for line in f.readlines():
            key, value = line.split('=')
            env[key] = value.split('\n')[0]
        return env


def export_env(env_path: str):
    """Loads your .env file and exports the three variables important for mlflow.
    MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_URI
    """
    env = load_env(env_path)
    for key in ["MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_URI"]:
        os.environ[key] = env[key]
