from setuptools import setup


setup(\
        name="symr", \
        packages=["symr", "symr_tests"], \
        version = "0.0", \
        description = "The Metric is the Massage", \
        install_requires=[\
                        "jupyter==1.0.0",\
                        "notebook>=6.4.12",\
                        "numpy>=1.21.6",\
                        "matplotlib==3.3.4",\
                        "scipy==1.5",\
                        "apted==1.0.3",\
                        "coverage==6.4.4",\
                        "scikit-learn==1.1.2",\
                        "sympy==1.10.1"]
    )


