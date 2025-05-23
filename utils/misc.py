import string
import random
from datetime import datetime
from omegaconf import DictConfig

from utils.smplx_utils import get_smplx_dimension_from_keys

def timestamp_str() -> str:
    """ Get current time stamp string
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")

def random_str(length: int=4) -> str:
    """ Generate random string with given length
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=4))

def compute_model_dim(cfg: DictConfig) -> int:
    """ Compute modeling dimension for different task

    Args:
        cfg: task configuration
    
    Return:
        The modeling dimension
    """
    if cfg.name == 'grasp_gen_ur':
        return 3 + 24
    else:
        raise Exception('Unsupported task.')

