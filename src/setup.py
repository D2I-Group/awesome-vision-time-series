from setuptools import setup, find_packages

VERSION = '0.0.3' 
DESCRIPTION = 'Time Series to Vision Trial Version'
LONG_DESCRIPTION = 'This is a package for converting time series data to vision data.'

# 配置
setup(
       # 名称必须匹配文件名 'verysimplemodule'
        name="timetovisiontrial", 
        version=VERSION,
        author="Ziming Zhao",
        author_email="zzhao41@cougarnet.uh.edu",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['pyts>=0.13.0', 'PyWavelets>=1.8.0'],
        
        keywords=['python', 'vision', 'time series', 'trial package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)