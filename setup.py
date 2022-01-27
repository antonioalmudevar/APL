from setuptools import setup

setup(  name='APL_mod',
        version='1.0',
        description='DAS for APL',
        author='Antonio Almudevar',
        author_email='almudevar@unizar.es',
        packages=['APL'],
        scripts=[
            'bin/AE',
            'bin/AE_dist',
            'bin/save_raw',
        ]
    )