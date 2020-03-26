from setuptools import setup, Extension

import versioneer

setup (name = 'sodetlib',
       description='SODETLIB',
       packages=['sodetlib'],
       scripts=['hammers/jackhammer'],
       version=versioneer.get_version(),
       cmdclass=versioneer.get_cmdclass())
