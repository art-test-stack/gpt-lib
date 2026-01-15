from pathlib import Path
import os
import random


def slugify(text: str) -> str:
    """Convert text to a slug suitable for filenames and URLs."""
    return "".join(c if c.isalnum() else "_" for c in text.replace(" ", "-")).lower()

def get_repo_dir():
    if os.getenv("GPT_LIB_BASE_DIR"):
        return Path(os.getenv("GPT_LIB_BASE_DIR"))
    
    else:
        home_dir = Path.home()
        cache_dir = home_dir / ".gpt_lib"
        repo_dir = cache_dir / "gpt_lib"
        return repo_dir
    
def print0(*values, **kwargs):
    """Print message only if on global rank 0"""
    rank = int(os.getenv("RANK", 0))
    if rank == 0:
        print(*values, **kwargs)

def print_banner():
    """Banner made with https://manytools.org/hacker-tools/ascii-banner/"""
    banner1 = """
      .-_'''-.   .-------. ,---------.   .---.    .-./`)  _______    
 '_( )_   \  \  _(`)_ \\          \  | ,_|    \ .-.')\  ____  \  
|(_ o _)|  ' | (_ o._)| `--.  ,---',-./  )    / `-' \| |    \ |  
. (_,_)/___| |  (_,_) /    |   \   \  '_ '`)   `-'`"`| |____/ /  
|  |  .-----.|   '-.-'     :_ _:    > (_)  )   .---. |   _ _ '.  
'  \  '-   .'|   |         (_I_)   (  .  .-'   |   | |  ( ' )  \ 
 \  `-'`   | |   |        (_(=)_)   `-'`-'|___ |   | | (_{;}_) | 
  \        / /   )         (_I_)     |        \|   | |  (_,_)  / 
   `'-...-'  `---'         '---'     `--------`'---' /_______.'  
                                                                 
"""
    banner2 = """
________________________________________________________________________
_/~~~~~~\__/~~~~~~~\__/~~~~~~~~\___________/~~\_______/~~~~\_/~~~~~~~\__
/~~\__/~~\_/~~\__/~~\____/~~\______________/~~\________/~~\__/~~\__/~~\_
/~~\_______/~~~~~~~\_____/~~\____/~~~~~~~\_/~~\________/~~\__/~~~~~~~\__
/~~\__/~~\_/~~\__________/~~\______________/~~\________/~~\__/~~\__/~~\_
_/~~~~~~~\_/~~\__________/~~\______________/~~~~~~~~\_/~~~~\_/~~~~~~~\__
______/~~\______________________________________________________________
"""
    banner = random.choice([banner1, banner2])
    print0(banner)
    return banner

class DummyWandb:
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def init(self, *args, **kwargs):
        pass
    def finish(self, *args, **kwargs):
        pass