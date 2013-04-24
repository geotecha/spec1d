from setuptools import setup
#maybe use:
#from distutils.core import setup
# why different? http://stackoverflow.com/questions/6344076/differences-between-distribute-distutils-setuptools-and-distutils2

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='spec1d',
      version='0.1',
      description='Multilayer vertical and radial soil consolidation using the spectral method',
      long_description=readme(),
      classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Geotechnical Engineering',
      ],
      keywords='funniest joke comedy flying circus',
      url='https://github.com/geotecha/spec1d.git',
      author='Rohan Walker',
      author_email='rtrwalker@gmail.com',
      license='GNU General Public License v3 or later (GPLv3+)',
      packages=['spec1d'],
      install_requires=[],
      zip_safe=False)