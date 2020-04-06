import funcs
import pickle

if __name__ == '__main__':
    with open('../app/data/func_feature2group_over_cycle_pay_to_group.pickle', 'wb') as f:
        pickle.dump(funcs.over_cycle_pay_to_group, f)
    with open('../app/data/func_feature2group_inq_main_to_group.pickle', 'wb') as f:
        pickle.dump(funcs.inq_main_to_group, f)
