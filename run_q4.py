"""独立第四问入口：python run_q4.py --experiments。"""
from pathlib import Path
from q4.runner import main

if __name__=='__main__':
    main(Path(__file__).resolve().parent)
