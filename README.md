# axcl-python-application-examples
A collection of python application demos and projects using the M5Stack LLM 8850 accelerator (Axera ax650) and rpi5. 
Running on debian Trixie.  

Required system packages: 
> opencv_python>=4.12.0.88  
> picamera2>=0.3.31

The repo contains an install.sh script to create a single venv that can perform all of the demos in this repo.   The script performs the following actions:
1. Checks for python, opencv, and picamera2 and install if not present (via sudo apt)
2. Create python virtual environment called "axcl_venv" that has access to system packages (for opencv, picam2 access)
3. Installs requirements.txt in the venv (via pip)
4. Creates a bash alias "axclenv" that will activate the environment from anywhere on your device (must close and reopen terminal or merge bashrc settings, see below)

To run the install script, navigate to repo then: 
``` 
chmod +x install.sh  
./install.sh  
```

Then close the terminal window or merge the updated bashrc settings with: 
```
source ~/.bashrc
```

In a new terminal window, type 'axclenv' to activate your venv.  

I've excluded models, weights etc but they can all be found at Axera-Tech HF page, linked below.  Model specific instructions are in each model's subdirectory.  

**LLM 8850 Documentation**  
https://docs.m5stack.com/en/guide/ai_accelerator/overview  
https://axcl-docs.readthedocs.io/zh-cn/latest/index.html  
https://huggingface.co/AXERA-TECH  
https://github.com/AXERA-TECH  
