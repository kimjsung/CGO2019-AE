#
#
#
def helper_base_find_list_of_list(list_of_list, target, location=0):
    #
    idx_count = 0
    for each_list in list_of_list:
        for each_element in each_list:
            if each_element == target:
                return idx_count
        #
        idx_count += 1
    #
    return -1

#
#   find value by using index from a list ("idx", value)
#
def helper_base_find_list_2D(list, index):
    for temp in list:
        if temp[0] == index:
            return temp[1]
    return -1

#
#
#
def helper_base_find_list_1D(list, index):
    for temp in list:
        if temp == index:
            return temp
    return -1