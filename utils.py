from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
def random_search_cv(model, X, y, param_grid, n_iter=100, cv=5, n_jobs=-1):
    """
    Perform random search cross-validation for a given model.
    
    Parameters:
    -----------
    model : estimator object
        The model to tune
    X : array-like
        Training data
    y : array-like
        Target values
    param_grid : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try
    n_iter : int, default=100
        Number of parameter settings sampled
    cv : int, default=5
        Number of cross-validation folds
    n_jobs : int, default=-1
        Number of jobs to run in parallel
        
    Returns:
    --------
    dict containing:
        - best_model: fitted model with best parameters
        - best_params: dictionary with best parameters
        - best_score: mean cross-validated score of best model
        - cv_results: full results
    """
    
    # Create scorer
    scorer = make_scorer(accuracy_score)
    
    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scorer,
        n_jobs=n_jobs,
        random_state=42,
        verbose=1
    )
    
    # Fit random search
    random_search.fit(X, y)
    
    # Print results
    print("\nBest parameters found:")
    print(random_search.best_params_)
    print(f"\nBest cross-validation accuracy: {random_search.best_score_:.4f}")
    
    return {
        'best_model': random_search.best_estimator_,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'cv_results': random_search.cv_results_
    }


def plot_search_results(cv_results, param_name=None, figsize=(12, 6)):
    """
    Plot the impact of a parameter on model performance.
    
    Parameters
    ----------
    cv_results : dict
        The cv_results_ dictionary from RandomizedSearchCV or GridSearchCV
    param_name : str, optional
        Name of the parameter to plot. If None, plots overall results
    figsize : tuple, default=(12, 6)
        Size of the figure in inches
    """
    plt.figure(figsize=figsize)
    
    if isinstance(cv_results, dict) and 'cv_results_' in cv_results:
        cv_results = cv_results['cv_results_']
    
    if param_name:
        try:
            # Extract parameter values and scores
            param_key = f'param_{param_name}'
            param_values = cv_results[param_key]
            mean_scores = cv_results['mean_test_score']
            std_scores = cv_results['std_test_score']
            
            # Convert to numpy arrays if needed
            param_values = np.array(param_values)
            mean_scores = np.array(mean_scores)
            std_scores = np.array(std_scores)
            
            # Sort values for better visualization
            sorted_idx = np.argsort(param_values)
            param_values = param_values[sorted_idx]
            mean_scores = mean_scores[sorted_idx]
            std_scores = std_scores[sorted_idx]
            
            # Plot mean scores with error bars
            plt.errorbar(param_values, mean_scores, yerr=std_scores, 
                        fmt='o-', capsize=5, capthick=1, elinewidth=1,
                        label='CV Score')
            
            # Add best score point
            best_idx = np.argmax(mean_scores)
            plt.scatter(param_values[best_idx], mean_scores[best_idx], 
                       color='red', s=100, label='Best Score', zorder=5)
            
            plt.xlabel(param_name.replace('_', ' ').title())
            
        except KeyError:
            print(f"Parameter '{param_name}' not found in CV results.")
            print("Available parameters:", [k.replace('param_', '') for k in cv_results.keys() 
                                         if k.startswith('param_')])
            return
    else:
        # Plot overall results
        mean_scores = cv_results['mean_test_score']
        std_scores = cv_results['std_test_score']
        iterations = range(len(mean_scores))
        
        plt.errorbar(iterations, mean_scores, yerr=std_scores,
                    fmt='o-', capsize=5, capthick=1, elinewidth=1,
                    label='CV Score')
        
        plt.xlabel('Iteration')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylabel('Mean CV Score (± std)')
    plt.title(f'Model Performance' + (f' vs {param_name}' if param_name else ''))
    plt.legend()
    plt.tight_layout()
    plt.show()