import setuptools

setuptools.setup(
    install_requires=[
        'google-cloud-aiplatform',
        'google-cloud-bigquery'
        'scikit-learn',
        'joblib'
    ],
    packages=setuptools.find_packages()
)
