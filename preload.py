import sys
import fire
sys.path.append('lib/')
from CGAN_runners import preload_ds

if __name__ == '__main__': 
    fire.Fire(preload_ds) 