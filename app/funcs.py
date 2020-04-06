import numpy as np
import woe.feature_process as fp


# define group, note this is an example for inadequate implementation, which later on will be corrected
def inq_main_to_group(array):
    conditions = [
        (array == 0),
        (array <= 9) & (array >= 1),
        (array <= 20) & (array >= 10),
        (array > 20)]
    choices = ['0 - missing','1-9','10-20','21 and +']
    return np.select(conditions, choices, default='null')


# define how overcycle payment amount should be grouped
# note this kind of grouping needs to be carefully examined to deal with missing values to make sure
# the implementation is correct
def over_cycle_pay_to_group(array):
    conditions = [
        (array <= -3816),
        (array <= -993) & (array > -3816),
        (array < 0) & (array > -993),
        (array == 0),
        (array > 0)]
    choices = ['<=-3816', '-3816 to -993', '-993 to 0', '0', '>0']
    return np.select(conditions, choices, default='null')


# define function to convert group variable to corresponding woe
def group_to_woe(df, group_var, global_bt, global_gt, min_sample=100, alpha=0.01):
    """
    This function is relying on woe package and will calculate woe for a grouped variable
    :param df: dataframe containing grouped variables and target variable
    :param group_var: group variable to be used for calculating woe
    :param global_bt: total number of bads
    :param global_gt: total number of goods
    :param min_sample: min volume of sample in one bin
    :param alpha: min IV improvement needed for re-binning
    :return: panda series containing woe value calcualted from the original group info
    """
    split = fp.binning_data_split(df, group_var, global_bt, global_gt, min_sample, alpha)
    civ = fp.format_iv_split(df, group_var, split.split_point, global_bt, global_gt)
    woe_dict = dict(zip(split.split_point, civ.woe_list))

    return df[group_var].map(woe_dict), woe_dict

