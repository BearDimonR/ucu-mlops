import setuptools

setuptools.setup(
    install_requires=[
        'google-cloud-aiplatform',
        'google-cloud-bigquery',
        'google-cloud-bigquery[pandas]',
        'db-dtypes',
        'scikit-learn',
        'joblib',
    ],
    packages=setuptools.find_packages()
)
