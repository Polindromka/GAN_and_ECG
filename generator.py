import sys
import fire
sys.path.append('lib/')
from CGAN_runners import display_gan

if __name__ == '__main__': 
    fire.Fire(display_gan) 