# Recommended tools
- Visual studio code 
- nvim with pyright 
- WSL or linux based OS 
- A git client to store your PAT if you are uncomfortable with the git CLI

# Prerequisites
- Note: Your *system* python version does *not* need to be 3.11, you just need a python 3.11 binary somewhere on your system to give to the virtual environment. You can build from source, or you can download precompiled binaries from the python website. If you have trouble you can ask Tristan for help getting the correct binary.
- If you are on windows, I highly recommend installing WSL 2 and using Ubuntu. It is your choice though.
- `python --version` needs to be at least 3.11
- `virtualenv` must be installed. If you are on linux, some distros have it in the distro's repo, if not you can use this link for help installing [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html)
- Create a Personal Access Token (PAT) to access the repo [how to create PAT](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens). Save this somewhere secure and easily accessible as you need it for all remote git operations. 
- If you have not already, you should enable 2FA on your github account so that your account does not become restricted. 

# Get started
- `git clone https://github.com/Team-69-MLIDS/ids_engine.git`
- `git checkout library`
- `cd ids_engine`
- `virtualenv -p <python path> venv` <python path> must point to a python 3.11 binary (eg. /usr/bin/python for linux users with their system python version @ 3.11)
- Linux only: `chmod +x ./venv/bin/activate` to make the venv activation script executable if it is not already
- Linux: `source ./venv/bin/activate`  Windows: `./venv/scripts/activate.ps1` to activate the virtual environment. Repeat this step each time you open the project in a new shell session. 
- `python --version` to ensure that the venv python version is 3.11.x, the minor patch version does not matter.
- `pip --version` to make sure pip is installed in the venv. If it is not, you can refer to [here](https://pip.pypa.io/en/stable/installation/) to install it in the venv.
- With an activated virtual environment shell, run `pip install -r requirements.txt` to install dependencies for the project

# Running the project
- `python -m flask --app src/server init-db` to initialize the database 
- `python -m flask --app src/server run` should start up a server.  
- `python src/ids/lccde/lccde_globecom.py` will run the LCCDE algorithm, you should try this so that you can get an idea of how the engine works and what its I/O is.



