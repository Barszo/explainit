from numpy import unique, array, all, empty_like
from ..logging_config import logger
from itertools import combinations

def get_combinations_for_modifiability(numerical_idx, categorical_sets, actionability, n_cols):
    """Generate combinations of indices based on actionability and modifiability."""
    
    # Filter indices based on actionability
    categorical_sets = [ cat_set for cat_set in categorical_sets if all(actionability[i] for i in cat_set)]
    numerical_idx = [i for i in numerical_idx if actionability[i]]

    actionable_indices = numerical_idx + categorical_sets

    # Generate combinations of the specified length
    result = []
    for elem in list(combinations(actionable_indices, n_cols)):
        if type(elem[0]) is list:
            result.append(tuple(elem[0]))
        else:
            result.append(tuple(elem))

    return result

def basic_function(
                X: list,
                actionability: list,
                categorical_groups: dict,
                model_predict,
                original_values: list,
                learning_rate: list,
                expected_prediction: list, 
                max_modifiability: int,
                max_examples: int,
                type_of_search: str = 'gradual'):
    """
    type_of_search: str can be 'gradual' or 'random'
    """

# * CATEGORICAL VALIDATION
    # Checking unique values in categorical columns for each category
    unique_categories_dict = {tuple(i_list): [unique(X[:, i_list], axis=0)] \
                        for cat, i_list in categorical_groups.items()}

    categorical_idx = []
    for elem in list(categorical_groups.values()):
        categorical_idx.extend(elem)

    numerical_idx = [i for i in range(len(X[0])) if i not in categorical_idx]

    logger.info(f"Categorical indices: {categorical_idx}")
    logger.info(f"Numerical indices: {numerical_idx}")

# * INITIAL SELECTION

    numerous_cols_dict = {}
    for n_cols in range(1,max_modifiability+1):
        modification_set = get_combinations_for_modifiability(numerical_idx, list(categorical_groups.values()), actionability, n_cols)

        logger.info(f"Number of combinations {len(modification_set)}")
        
        mod_vals_dict = {}
        for cols_idx in modification_set:

            # * If only one column is selected for modification
            if n_cols == 1:

                # ! Categorical features
                if tuple(cols_idx) in list(unique_categories_dict.keys()):

                    # finding set of unique categorical values and removing
                    # the original value from the set
                    cats_set = unique_categories_dict[tuple(cols_idx)][0]  # Extract from list
                    original_cat = array([original_values[i] for i in cols_idx])
                    mask = ~all(cats_set == original_cat, axis=1)
                    cats_set = cats_set[mask]

                    mod_vals_list = []
                    for cats in cats_set:
                        mod_values = original_values.copy()  # Start with original values
                        for i, idx in enumerate(cols_idx):
                            mod_values[idx] = cats[i]
                            # print(model_predict([mod_values]), expected_prediction)
                            if model_predict([mod_values]) == expected_prediction:
                                mod_vals_list.append(mod_values)

                # ! Numerical features
                # TODO: Modify learning rate based on probability, if change did not 
                # lead to high probability then learning rate should be increased
                # or decreased accordingly
                else:
                    
                    lr = learning_rate[cols_idx[0]]
                    if type_of_search == 'gradual':
                        mod_vals_list = []

                        # Looking higher than original value
                        for i in range(0, max_examples + 1):
                            mod_values = original_values.copy()
                            mod_values[cols_idx[0]] += i * lr
                            
                            if model_predict([mod_values]) == expected_prediction:
                                mod_vals_list.append(mod_values)
                                break

                        # Looking lower than original value
                        for i in range(0, max_examples + 1):
                            mod_values = original_values.copy()
                            mod_values[cols_idx[0]] -= i * lr
                            
                            if model_predict([mod_values]) == expected_prediction:
                                mod_vals_list.append(mod_values)
                                break

                mod_vals_dict[cols_idx] = mod_vals_list

            # TODO: handle case with multiple columns selected for modification
            else:
                pass

            numerous_cols_dict[n_cols] = mod_vals_dict
        print(mod_vals_dict)

            # Modify categorical columns with unique_categories_dict



# * SHAP EVALUATION

# * FINAL SELECTION


    # return unique_rows