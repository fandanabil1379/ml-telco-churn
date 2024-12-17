import numpy as np  
import pandas as pd 
import seaborn as sns   
import scipy.stats as stats
import matplotlib.pyplot as plt 
from scipy.stats import chi2_contingency, mannwhitneyu   
from sklearn.metrics import fbeta_score, multilabel_confusion_matrix

# ---HELP FUNCTION---

def plot_bar_churn_comparison(df, var_col, churn_col, palette=None):
    """
    Visualizes churn rate and churn count by a variable with annotations.

    Parameters:
    - data (pd.DataFrame): Input dataframe containing gender and churn columns.
    - var_col (str): Column name for object.
    - churn_col (str): Column name for churn (with values 'Yes' and 'No').
    - palette (dict): Custom color palette.

    Returns:
        None: Displays the plots using matplotlib.

    """
    # Set default color palette if none is provided
    if palette is None:
        palette = {'Yes': '#007A33', 'No': '#A0A0A0'}  # BCG palette
    
    # Aggregate churn count for each category in the specified variable
    churn_count = df.groupby([var_col, churn_col]).size().reset_index(name='Count')
    
    # Calculate churn rate (percentage of 'Yes' churn) for each category
    churn_rate = (
        df.groupby(var_col)[churn_col]
        .value_counts(normalize=True)  # Normalize to calculate proportion
        .rename("Rate")  # Rename normalized count column
        .reset_index()
    )
    churn_rate = churn_rate[churn_rate[churn_col] == 'Yes']  # Filter only 'Yes' churn rates

    # Determine colors: green for the highest churn rate, gray for others
    max_rate_index = churn_rate['Rate'].idxmax()
    churn_rate['Color'] = ['#008000' if i == max_rate_index else '#A0A0A0' for i in churn_rate.index]

    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Churn Analysis by {var_col.title()}", fontsize=14, fontweight='bold', color='#007A33')

    # Plot churn count on the first subplot
    sns.barplot(
        data=churn_count, x=var_col, y='Count', hue=churn_col, ax=axes[0], palette=palette, width=0.5
    )
    axes[0].set_title(f"Churn Count by {var_col.title()}", fontsize=14)  # Set plot title
    axes[0].set_ylabel("Count", fontsize=12)  # Label for y-axis
    axes[0].set_xlabel(None)  # Remove x-axis label for aesthetics
    axes[0].set_facecolor("none")  # Set subplot background transparent
    axes[0].get_yaxis().set_visible(False)  # Hide y-axis ticks
    axes[0].spines[['left', 'top', 'bottom']].set_visible(False)  # Hide left spine for minimalistic look
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%d', label_type='edge', fontsize=10)  # Annotate bar values

    # Add a custom legend for first subplot
    legend1 = [
        plt.Line2D([0], [0], color='#008000', lw=4, label='Churn'),
        plt.Line2D([0], [0], color='#A0A0A0', lw=4, label='Not Churn')
    ]
    axes[0].legend(handles=legend1, fontsize=10, title_fontsize=12, loc='best')

    # Plot churn rate on the second subplot
    sns.barplot(
        data=churn_rate, x=var_col, y='Rate', ax=axes[1], palette=churn_rate['Color'], width=0.3
    )
    axes[1].set_title(f"Churn Rate by {var_col.title()}", fontsize=14)  # Set plot title
    axes[1].set_ylabel("Churn Rate", fontsize=12)  # Label for y-axis
    axes[1].set_xlabel(None)  # Remove x-axis label for aesthetics
    axes[1].set_facecolor("none")  # Set subplot background transparent
    axes[1].get_yaxis().set_visible(False)  # Hide y-axis ticks
    axes[1].spines[['left', 'top', 'bottom', 'right']].set_visible(False)  # Hide all spine for minimalistic look
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.2f', label_type='edge', fontsize=10)  # Annotate bar values

    # Add a custom legend for second subplot
    legend2 = [
        plt.Line2D([2], [0], color='#008000', lw=4, label='Highest Churn Rate'),
        plt.Line2D([2], [0], color='#A0A0A0', lw=4, label='Other Categories')
    ]
    axes[1].legend(handles=legend2, fontsize=10, title_fontsize=12, loc='best')
    
    # Adjust subplot layout to avoid overlapping titles and labels
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_boxplot_churn_comparison(df, var_col, churn_col, palette=None):
    """
    Visualizes the distribution of a numerical variable by churn status using a boxplot.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        var_col (str): The column name of the numerical variable to analyze.
        churn_col (str): The column name indicating churn status ('Yes' or 'No').
        palette (dict, optional): A dictionary specifying the color palette for the churn categories.

    Returns:
        None: The function displays a boxplot.
    """
    # Set default color palette if none is provided
    if palette is None:
        palette = {'Yes': '#007A33', 'No': '#A0A0A0'}  # BCG palette

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(
        f"Churn Analysis by {var_col.title()}",
        fontsize=13,
        fontweight='bold',
        color='#007A33'
    )  # Add a title for the plot

    # Create a boxplot to visualize the distribution of the variable by churn status
    sns.boxplot(
        x=churn_col, y=var_col, data=df, palette=palette, width=0.3
    )
    
    # Customize the x-axis labels for better readability
    ax.set_xticklabels(['Not Churn', 'Churn'], fontsize=12)

    # Adjust visual aesthetics for a clean and minimalistic look
    ax.set_facecolor('none')  # Set the subplot background to transparent
    ax.spines[['left', 'top', 'bottom', 'right']].set_visible(False)  # Hide the all spine for a cleaner look
    ax.set_xlabel(None)  # Remove the x-axis label
    ax.set_ylabel(None)  # Remove the y-axis label
    ax.grid(False)

    # Adjust layout to prevent overlapping elements and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def chi2_with_effect_size(df, var_col, churn_col):
    """
    Perform a Chi-Square test and calculate Cramér's V to analyze the relationship between a categorical variable and churn.

    Parameters:
        df (pd.DataFrame): The dataset containing the relevant columns.
        var_col (str): The column name of the categorical variable to be tested.
        churn_col (str): The column name representing the churn status (e.g., 'Yes' or 'No').

    Returns:
        None: Prints the Chi-Square statistic, p-value, degrees of freedom, expected frequencies, 
        and Cramér's V effect size interpretation.
    """
    # Create a contingency table between the churn column and the variable of interest
    contingency_table = pd.crosstab(df[churn_col], df[var_col])

    # Perform the Chi-Square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # Perform the Cramér's V
    n = contingency_table.sum().sum()  # Jumlah total observasi
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

    # Print the Chi-Square results
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-Value: {p_value:.4f}")
    #print(f"Degrees of Freedom: {dof:.4f}")

    # Interpret the Cramér's V effect size
    if cramers_v < 0.1:
        print("Effect size: Negligible")  # Weak or no association
    elif cramers_v < 0.3:
        print("Effect size: Small")  # Small association
    elif cramers_v < 0.5:
        print("Effect size: Medium")  # Medium association
    else:
        print("Effect size: Large")  # Strong association

def mannu_with_effect_size(df, var_col, churn_col):
    """
    Perform Mann-Whitney U Test to compare the distribution of a variable between churned and non-churned customers
    and calculate the effect size (r).

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    var_col (str): The column name for the variable to be tested.
    churn_col (str): The column name indicating churn status ('Yes' or 'No').

    Returns:
    None: Prints the results of the test, including the p-value, U statistic, and effect size (r).
    """
    # Select data for churned and non-churned customers
    churn_data = df[df[churn_col] == 'Yes'][var_col]  # Data for customers who churned
    not_churn_data = df[df[churn_col] == 'No'][var_col]  # Data for customers who did not churn
    
    # Perform the Mann-Whitney U Test to compare distributions
    u_statistic, p_value = mannwhitneyu(churn_data, not_churn_data)  # Statistical test
    
    # Calculate Z value from the p-value (two-sided test)
    try:
        z = stats.norm.ppf(1 - p_value / 2)
    except Exception:
        z = np.inf  # Handle calculation errors

    # Calculate effect size (r) using the Z score from the p-value
    n1 = len(churn_data)  # Number of churned customers
    n2 = len(not_churn_data)  # Number of non-churned customers
    try:
        effect_size = z / np.sqrt(n1 + n2)  # Effect size (r)
        if not np.isfinite(effect_size):
            effect_size = np.nan
    except ZeroDivisionError:
        effect_size = np.nan

    # Print the Chi-Square results
    print(f"Mann-Whitney U Statistic: {u_statistic:.4f}")
    print(f"P-Value: {p_value:.4f}")

    # Interpretation of effect size
    if np.isnan(effect_size):
        print("Effect size cannot be computed.")
    elif effect_size < 0.1:
        print("Effect size: Negligible effect")
    elif effect_size < 0.3:
        print("Effect size: Small effect")
    elif effect_size < 0.5:
        print("Effect size: Medium effect")
    else:
        print("Effect size: Large effect")


def total_cost(y_true, y_pred):
    """
    Calculate the total cost associated with false positives (FP) and false negatives (FN) in a binary classification model.

    Parameters:
        y_true (numpy.ndarray or list): Ground truth binary labels (0 or 1).
        y_pred (numpy.ndarray or list): Predicted binary labels (0 or 1).

    Returns:
        float: The total cost associated with the predictions.
    """
    # Calculate the number of false positives (FP)
    FP = np.sum((y_pred == 1) & (y_true == 0))
    
    # Calculate the number of false negatives (FN)
    FN = np.sum((y_pred == 0) & (y_true == 1)) 
    
    # Calculate the total cost using the given cost coefficients
    total_cost = (FP * 138.8) + (FN * 694)
    return total_cost


def show_predict(estimator, sample):
    """
    Predict the churn outcome for a given sample and display the probabilities.

    Parameters:
        estimator: Trained model that implements `predict` and `predict_proba` methods.
        sample (pd.DataFrame or similar): Single sample (as a DataFrame or similar format) to predict.

    Returns:
        pd.DataFrame or dict: Modified sample with churn prediction (`Churn`) and probabilities (`proba_0`, `proba_1`).
    """
    # Predict the original outcome for the sample
    print(f'Original Outcome: {estimator.predict(sample)[0]}')

    # Predict probabilities for each class
    proba = estimator.predict_proba(sample)

    # Copy sample data and append prediction details
    outcome = sample.copy()
    if proba[0][0] > proba[0][1]:
        outcome['Churn'] = 0  # Class 0 (not churn)
    else:
        outcome['Churn'] = 1  # Class 1 (churn)
    
    # Add rounded probabilities to the output
    outcome['proba_0'] = np.round(proba[0][0], 2)  # Probability of Class 0
    outcome['proba_1'] = np.round(proba[0][1], 2)  # Probability of Class 1
    return outcome

def compute_confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix for multilabel classification and return the metrics for a specific class.

    Parameters:
        y_true (np.array or list): True labels.
        y_pred (np.array or list): Predicted labels.

    Returns:
        tuple: True negatives (tn), false positives (fp), false negatives (fn), true positives (tp) for the specific class.
    """
    # Compute multilabel confusion matrices for all classes
    multi_cm = multilabel_confusion_matrix(y_true, y_pred)

    # Specify the class of interest (e.g., class 1)
    class_index = 1  # This can be adjusted to target another class
    tn, fp, fn, tp = multi_cm[class_index].ravel()
    return tn, fp, fn, tp

def compute_result(data_with_labels, name):
    """
    Compute performance metrics for a churn prediction experiment.

    Parameters:
        data_with_labels (pd.DataFrame): Data containing true labels (`Churn`) and predicted labels (`prediction_label`).
        name (str): Experiment name or identifier.

    Returns:
        pd.DataFrame: DataFrame with experiment name, false positives (FP), false negatives (FN),
                      F5-score, and total cost of errors.
    """
    # Extract true and predicted labels
    ytrue = data_with_labels['Churn']
    ypred = data_with_labels['prediction_label']

    # Calculate the total cost of errors
    tc = total_cost(ytrue, ypred)

    # Calculate F5-score (gives higher weight to recall)
    f4 = fbeta_score(ytrue, ypred, beta=5)

    # Compute confusion matrix metrics
    tn, fp, fn, tp = compute_confusion_matrix(ytrue, ypred)

    # Return a summary of results as a DataFrame
    return pd.DataFrame(
        {'Experiment': [name], 'FP': [fp], 'FN': [fn], 'F5': [f4], 'Loss (in USD)': [tc]}
    )