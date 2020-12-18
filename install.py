import importlib
import pip

# List of packages and respective library name
package_libraryName = {"numpy":"numpy==1.19.3",\
                       "cv2":"opencv-python"}

# Install module if not available
def install(module):
    
    # Check if module is installed
    not_found = importlib.util.find_spec(module) is None
    
    # if not, install
    if not_found:
        
        # Get library name
        if module in package_libraryName:
            libraryName = package_libraryName[module]
        else:
            libraryName = module
        
        # Install module
        print("libraryName '" + libraryName + \
              "' for import '" + module + "' was not found! Trying to install it.")
        pip.main(['install', libraryName])
