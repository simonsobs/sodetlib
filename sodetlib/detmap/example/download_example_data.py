import os
import shutil
import zipfile
import gdown

real_path_this_file = os.path.dirname(os.path.realpath(__file__))
abs_path_sodetlib, _ = real_path_this_file.rsplit("sodetlib", 1)
abs_path_detmap = os.path.join(abs_path_sodetlib, "sodetlib", "detmap")
zip_file_id_sample_data = '1G8eiJ85zVKu53GCzeVdjHEV8cWqrg6eH'


def sample_data_init(del_dir=False, zip_file_id=zip_file_id_sample_data, folder_name='sample_data'):
    zipfile_url = f'https://drive.google.com/file/d/{zip_file_id_sample_data}/view?usp=sharing'
    zipfile_path = os.path.join(abs_path_detmap, f'{folder_name}.zip')
    data_dir = os.path.join(abs_path_detmap, f'{folder_name}')
    # get the absolute path fo the detector mapping code
    if os.path.exists(data_dir):
        if del_dir:
            # Delete the data directory and downloaded again.
            print(f"Deleting the sample data dir: {data_dir}")
            shutil.rmtree(data_dir)
            print('Directory deleted')
        else:
            # the data exists and does not need to be downloaded again.
            return
    # Download the example time stream data.
    print(f'The {folder_name} is not found, doing a one time download of the {folder_name}')
    if os.path.exists(zipfile_path):
        # delete an old (failed) zip file if one is present.
        os.remove(zipfile_path)
    # start the download and tell the user what is happening
    print(f'  Beginning file download of at {zipfile_url}')
    print(f'  This is ~350 Mb file so it make take a while depending on your connection speed...')
    gdown.download(url=f'https://drive.google.com/uc?id={zip_file_id}', output=zipfile_path)
    print('Download Complete.\n')

    # unpack the zip file
    print("Extracting the zipfile...")
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(abs_path_detmap)
    # delete the zip ile now that it is unpacked
    os.remove(zipfile_path)
    print("  The zipfile extracted and unpacked.")
    print("  A cleanup was done remove original zipfile\n")
    print(f"Data unpacked at: {data_dir}")
    return


if __name__ == "__main__":
    sample_data_init(del_dir=True)

