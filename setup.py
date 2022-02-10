import os
from setuptools import setup, Extension, find_packages

import versioneer

# local metadata path
local_metadata = os.path.join('sodetlib', 'detmap', 'meta')
local_examples_dir = os.path.join('sodetlib', 'detmap', 'example')
meta_files_all = [os.path.join(local_metadata, f) for f in os.listdir(local_metadata)
                  if f[-4:].lower() in {'.csv', '.pkl'}]

setup(name='sodetlib',
      description='SODETLIB',
      packages=find_packages(),
      scripts=['hammers/jackhammer'],
      data_files=[(local_metadata, meta_files_all),
                  (local_examples_dir, [os.path.join(local_examples_dir, 'example.yaml')])],
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass())
