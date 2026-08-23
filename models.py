"""
Fog Top Height Retrieval Models Module
Regression and Classification Modeling Functions

Contains:
- Logistic regression model
- SVM regression model
- CALIPSO trajectory regression analysis
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import svm
from sklearn.metrics import r2_score, mean_squared_error


def logistic_regression2(refs, island_h, category):
    """
    Logistic regression model
    
    Args:
        refs (np.array): Reflectance array
        island_h (np.array): Island height array
        category (np.array): Island category array
        
    Returns:
        tuple: (a, b) - Regression coefficients
    """
    # Check if category 1 is empty
    if len(island_h[category == 1]) == 0:
        a = 0
        b = island_h[category == 2].max()
        return a, b
    # Check if category 2 is empty, if empty then height2 = height1
    if len(island_h[category == 2]) == 0:
        a = 0
        b = island_h[category == 1].min()
        return a, b

    clf = LogisticRegression(C=50, fit_intercept=False)
    clf = clf.fit(np.column_stack((refs, island_h)), category)
    w = clf.coef_[0]
    intercept = clf.intercept_
    a = -w[0]/w[1]
    b = -intercept / w[1]
    return a, b

def logistic_regression(refs, island_h, category):
    """
    Logistic regression model
    
    Args:
        refs (np.array): Reflectance array
        island_h (np.array): Island height array
        category (np.array): Island category array
        
    Returns:
        tuple: (a, b) - Regression coefficients
    """
    # Check if category 1 is empty
    if len(island_h[category == 1]) == 0:
        a = 0
        b = island_h[category == 2].max()
        return a, b
    # Check if category 2 is empty, if empty then height2 = height1
    if len(island_h[category == 2]) == 0:
        a = 0
        b = island_h[category == 1].min()
        return a, b

    clf = LogisticRegression(C=50)
    clf = clf.fit(np.column_stack((refs, island_h)), category)
    w = clf.coef_[0]
    intercept = clf.intercept_
    a = -w[0]/w[1]
    b = -intercept / w[1]
    return a, b

def svm_regression(refs, island_h, category):
    """
    SVM regression model
    
    Args:
        refs (np.array): Reflectance array
        island_h (np.array): Island height array
        category (np.array): Island category array
        
    Returns:
        tuple: (a, b) - Regression coefficients
    """
    # Check if category 1 is empty
    if len(island_h[category == 1]) == 0:
        a = 0
        b = island_h[category == 2].max()
        return a, b
    # Check if category 2 is empty, if empty then height2 = height1
    if len(island_h[category == 2]) == 0:
        a = 0
        b = island_h[category == 1].min()
        return a, b
    
    clf = svm.SVC(kernel='linear', C=50, gamma='auto')
    clf.fit(np.column_stack((refs, island_h)), category)
    w = clf.coef_[0]
    intercept = clf.intercept_
    a = -w[0]/w[1]
    b = -intercept / w[1]
    return a, b


def calipso_track_regression(calipso_cth, track_refs):
    """
    Perform linear regression analysis using calipso_cth and track_refs
    
    Parameters:
    calipso_cth (np.array): CALIPSO cloud top height data, shape (n, 3), where the 3rd column contains height values
    track_refs (np.array): Reflectance values corresponding to track points, shape (n,)
    
    Returns:
    dict: Dictionary containing regression coefficients, statistical metrics, and predicted values
    """
    # Extract valid data points (height > 0)
    if calipso_cth.ndim > 1:
        valid_mask = calipso_cth[:, 2] > 0
        cth_values = calipso_cth[valid_mask, 2]  # Cloud top height
        ref_values = track_refs[valid_mask]      # Reflectance values
    else:
        valid_mask = calipso_cth > 0
        cth_values = calipso_cth
        ref_values = track_refs
    
    # Ensure sufficient data points for regression
    if len(cth_values) < 2:
        return {
            'slope': 0,
            'intercept': 0,
            'r2': 0,
            'rmse': 0,
            'correlation': 0,
            'predicted': np.zeros_like(track_refs),
            'valid_count': len(cth_values)
        }
    
    # Perform linear regression: CTH = slope * Ref + intercept
    # Reshape data for sklearn
    X = ref_values.reshape(-1, 1)
    y = cth_values
    
    # Create and train linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get regression coefficients
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Calculate predicted values
    predicted = model.predict(X)
    
    # Calculate statistical metrics
    r2 = r2_score(y, predicted)
    rmse = np.sqrt(mean_squared_error(y, predicted))
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(ref_values, cth_values)[0, 1] * 100
    
    # Generate predicted values for all track_refs
    all_predicted = np.zeros_like(track_refs)
    all_predicted[valid_mask] = predicted
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r2': r2,
        'rmse': rmse,
        'correlation': correlation,
        'predicted': all_predicted,
        'valid_count': len(cth_values)
    }
