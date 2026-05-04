import warnings
import os

# Suppress scikit-learn joblib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.parallel')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')

# Set environment variable to reduce verbosity
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
