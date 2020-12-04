import pip

def install(package):
    try:
        __import__(package)
    except ImportError:
        if (package == "numpy"):
            package = "numpy==1.19.3"
        
        print("Package '" + package + "' not found! Trying to install it.")
        pip.main(['install', package])