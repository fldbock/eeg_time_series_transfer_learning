from urllib import request
from bs4 import BeautifulSoup
import json


# Instantiate url variable with download page url
base_url = "https://www.isip.piconepress.com"
url = "https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_slowing/v1.0.1/edf/"

# Define password manager and authentication handler to log in to TUH EEG Corpus download page
password_mgr = request.HTTPPasswordMgrWithDefaultRealm()
password_mgr.add_password(None, url, "nedc", "nedc_resources")
auth_handler = request.HTTPBasicAuthHandler(password_mgr)


# Recursively map out the structure of the directories on the download page
def get_dir_structure(dir_url, dir_name):
    # Open directory url with defined authentication handler
    dir_url_opener = request.build_opener(auth_handler)
    dir_url_bytes = dir_url_opener.open(dir_url) 
    
    try:
        dir_structure_dict = {}
        
        dir_html = BeautifulSoup(dir_url_bytes.read(), "lxml")
        child_url_tags = dir_html.find_all("a")
        
        for child_url_tag in child_url_tags:
            child_url = child_url_tag.get("href")
            
            # Filter out navigational elements            
            if (child_url[0] == "?" or url.find(child_url) > -1 or dir_url.find(child_url) > -1 or child_url.find(".lbl") > -1  or child_url.find("mailto:") > -1):
                continue

            if child_url[:-4] not in dir_structure and child_url[:-1] not in dir_structure:
                print(child_url)

            # Differentiate between .edf file, subfolder and other files
            if child_url[-4:] == ".edf" or child_url[-4:] == ".tse":
                # Url to .edf file
                file_name = child_url[:-4]
                file_url = dir_url + child_url

                if file_name not in dir_structure_dict:
                    dir_structure_dict[file_name] = file_url
                    
            elif child_url[-1] == "/":
                # Url to folder
                folder_name = child_url[:-1]
                if child_url[0] != "/":
                    # Url is extension
                    folder_url = dir_url + child_url
                else:
                    # Url is full path
                    folder_url = base_url + child_url
                
                if (folder_name not in dir_structure) or folder_name == "eval" or folder_name == "train":
                    dir_structure_dict[folder_name] = get_dir_structure(folder_url, folder_name)
            else:
                # Other file that is not relevant
                pass
    except Exception as e:
        print(e)
        if str(e) == "<urlopen error [Errno 60] Operation timed out>":
            dir_structure.update(dir_structure_dict)
            dir_structure_dict = get_dir_structure(dir_url, dir_name)
    finally:
        # Close opener object
        dir_url_bytes.close()
        return dir_structure_dict


with open('/home/jupyter/time_series_transfer_learning/data/JSON_files/TUSL/TUSL.json', 'r') as f:
    dir_structure = json.load(f)

# Instantiate url variable with download page url
temp_dir_structure = get_dir_structure(url, "TUSL")
#dir_structure.update(temp_dir_structure)

with open('/home/jupyter/time_series_transfer_learning/data/JSON_files/TUSL/TUSL.json', 'w') as f:
    json.dump(temp_dir_structure, f)