import sys
import fire
sys.path.append('lib/')
from model_new import test_model

if __name__ == '__main__': 
    fire.Fire(test_model) 