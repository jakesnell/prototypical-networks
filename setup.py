from setuptools import setup

setup(name='protonets',
      version='0.0.1',
      author='Jake Snell',
      author_email='jsnell@cs.toronto.edu',
      license='MIT',
      packages=['protonets'],
      install_requires=[
          'torch',
          'tqdm'
      ])
