# Facial Recognition Software

INSTALLATION:
----------------------------------
Make sure you have Python 3.6.8 (x64 bit) Installed

Open the folder where you downloaded FacialRecognitionSoftware-master.zip and unzip the folder.

Open Command Promt (CMD) in that folder.

Install Virtualenv Module in Python using command:
- pip install virtualenv

Now create a virtual environment folder using:
- virtualenv venv

This will take you to the vitual environment folder which is created right now.
- cd venv

Activate the virtual environment using:
- scripts\activate

Create new folder inside venv called "apps"

Extract all the ZIP app code files inside "apps" folder.

Now install all required modules using commands:
- cd apps
- pip install -r requirements.txt

Check for manage.py file and cd to that directory or folder.

Now run this command to run the Face recognition App:
- python manage.py runserver

Once it runs successfully now you can open you browser and type this URL in Browser:
- 127.0.0.1:8000

Now you will be inside the Web app and you can test it as you want.
-----------------------------------

How To Start Again After Installation:

Open CMD in "venv" folder
- scripts\activate
- cd apps
- python manage.py runserver
