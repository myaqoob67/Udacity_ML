
def remove_outlier(enron_data, keys):
    """ removes a list of keys from a dict object """
    for key in keys:
        enron_data.pop(key, 0)