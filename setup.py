from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='tfkit',
    version='0.8.5',
    description='Transformers kit - Multi-task QA/Tagging/Multi-label Multi-Class Classification/Generation with BERT/ALBERT/T5/BERT',
    url='https://github.com/voidful/TFkit',
    author='Voidful',
    author_email='voidful.stack@gmail.com',
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    setup_requires=['setuptools-git'],
    classifiers=[
        'Development Status :: 4 - Beta',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3.6'
    ],
    license="Apache",
    keywords='transformer huggingface nlp multi-task multi-class multi-label classification generation tagging deep learning machine reading',
    packages=find_packages(),
    install_requires=required,
    entry_points={
        'console_scripts': ['tfkit-train=tfkit.train:main', 'tfkit-eval=tfkit.eval:main', 'tfkit-dump=tfkit.dump:main']
    },
    py_modules=['tfkit'],
    python_requires=">=3.5.0",
    zip_safe=False,
)
