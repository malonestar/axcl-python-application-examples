# axcl-python-application-examples
A collection of python application demos and projects using the M5Stack LLM 8850 accelerator (Axera ax650) and rpi5. 

Required system packages: 
> numpy>=2.3.4  
> opencv_python>=4.12.0.88  
> picamera2>=0.3.31

The 'yolo' directory contains an install.sh script.  
This creates and activates a python venv 'axcl_venv' and installs the dependenies  
for the YOLO python applications.  

The install script also creates a bash alias 'axclenv' that when typed in a terminal window  
will activate your venv from anywhere on the machine.  

To run clone and navigate to the repo.  Then: 
```
cd yolo  
chmod +x install.sh  
./install.sh  
```

Then close the terminal window or merge the updated bashrc settings with: 
```
source ~/.bashrc
```

Now in a new terminal window, type 'axclenv' to activate your venv from anywhere.  

See README.md in 'yolo' directory for application specific details. 

**LLM 8850 Documentation**  
https://docs.m5stack.com/en/guide/ai_accelerator/overview  
https://axcl-docs.readthedocs.io/zh-cn/latest/index.html  
https://huggingface.co/AXERA-TECH  
https://github.com/AXERA-TECH  
