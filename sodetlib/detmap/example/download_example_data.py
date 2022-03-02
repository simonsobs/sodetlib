import os
import shutil
import zipfile
import requests


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_file_from_google_drive(id, destination):
    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(url, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
    save_response_content(response, destination)


def sample_data_init(del_dir=False):
    # get the absolute path fo the detector mapping code
    real_path_this_file = os.path.dirname(os.path.realpath(__file__))
    abs_path_sodetlib, _ = real_path_this_file.rsplit("sodetlib", 1)
    abs_path_detmap = os.path.join(abs_path_sodetlib, "sodetlib", "detmap")
    sample_data_dir = os.path.join(abs_path_detmap, 'sample_data')
    if os.path.exists(sample_data_dir):
        if del_dir:
            # Delete the data directory and downloaded again.
            print(f"Deleting the sample data dir: {sample_data_dir}")
            shutil.rmtree(sample_data_dir)
            print('Directory deleted')
        else:
            # the data exists and does not need to be downloaded again.
            return
    # Download the example time stream data.
    print('The Sample Data used for the default data not found, doing a one time download of the sample data data.')
    zip_file_id = '1G8eiJ85zVKu53GCzeVdjHEV8cWqrg6eH'
    zipfile_url = f'https://drive.google.com/file/d/{zip_file_id}/view?usp=sharing'
    zipfile_path = os.path.join(abs_path_detmap, 'sample_data.zip')
    if os.path.exists(zipfile_path):
        # delete an old (failed) zip file if one is present.
        os.remove(zipfile_path)
    # start the download and tell the user what is happening
    print(f'  Beginning file download of at {zipfile_url}')
    print(f'  This is ~350 Mb file so it make take a while depending on your connection speed...')
    download_file_from_google_drive(id=zip_file_id, destination=zipfile_path)
    print('Download Complete.\n')

    # unpack the zip file
    print("Extracting the zipfile...")
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(abs_path_detmap)
    # delete the zip ile now that it is unpacked
    os.remove(zipfile_path)
    print("  The zipfile extracted and unpacked.")
    print("  A cleanup was done remove original zipfile\n")
    print(f"Data unpacked at: {sample_data_dir}")
    return


if __name__ == "__main__":
    sample_data_init(del_dir=True)
