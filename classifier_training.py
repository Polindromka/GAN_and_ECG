import sys
import fire
sys.path.append('lib/')
from model_new import train_nn

if __name__ == '__main__': 
    fire.Fire(train_nn) 