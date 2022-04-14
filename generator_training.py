import sys
import fire
sys.path.append('lib/')
from CGAN_runners import run_params

if __name__ == '__main__': 
        fire.Fire(run_params)
