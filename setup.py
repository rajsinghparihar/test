import subprocess
import os
import time
os.popen("git clone https://github.com/CompVis/latent-diffusion.git")
os.popen("git clone https://github.com/CompVis/taming-transformers.git")
# subprocess.sleep(10)
time.sleep(10)
subprocess.run(["pip",  "install",  "-r",  "requirements.txt"])