# general
import glob, os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import pickle
import json
from copy import deepcopy

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
sns.set_style('darkgrid')

# warnings
import warnings
import logging

logging.basicConfig(filename= 'neurobot_logging.log', level = logging.DEBUG)
logger = logging.getLogger()
logger.info('The NEUROBOT started.')
