from setuptools import setup, Extension, find_packages

import versioneer

setup (name = 'sodetlib',
       description='SODETLIB',
       packages=find_packages(),
       scripts=['hammers/jackhammer'],
       version=versioneer.get_version(),
       cmdclass=versioneer.get_cmdclass())
