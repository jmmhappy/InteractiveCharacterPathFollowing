import zipfile
from urllib.request import urlretrieve

lafan1_url = 'https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip'
output_path = './lafan1.zip'

urlretrieve(lafan1_url, output_path)
with zipfile.ZipFile(output_path, 'r') as archive:
    archive.extractall('.')
