import logging 
import pandas as pd
import _pickle as cPickle

logger = logging.getLogger()

def create_vocab(train_file, user_vocab, item_vocab, cate_vocab):

    f_train = open(train_file, "r")

    user_dict = {}
    item_dict = {}
    cat_dict = {}

    logger.info("vocab generating...")
    for line in f_train:
        arr = line.strip("\n").split("\t")
        uid = arr[1]
        mid = arr[2]
        cat = arr[3]
        mid_list = arr[5]
        cat_list = arr[6]

        if uid not in user_dict:
            user_dict[uid] = 0
        user_dict[uid] += 1
        if mid not in item_dict:
            item_dict[mid] = 0
        item_dict[mid] += 1
        if cat not in cat_dict:
            cat_dict[cat] = 0
        cat_dict[cat] += 1
        if len(mid_list) == 0:
            continue
        for m in mid_list.split(","):
            if m not in item_dict:
                item_dict[m] = 0
            item_dict[m] += 1
        for c in cat_list.split(","):
            if c not in cat_dict:
                cat_dict[c] = 0
            cat_dict[c] += 1

    sorted_user_dict = sorted(user_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_item_dict = sorted(item_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_cat_dict = sorted(cat_dict.items(), key=lambda x: x[1], reverse=True)

    uid_voc = {}
    index = 0
    for key, value in sorted_user_dict:
        uid_voc[key] = index
        index += 1

    mid_voc = {}
    mid_voc["default_mid"] = 0
    index = 1
    for key, value in sorted_item_dict:
        mid_voc[key] = index
        index += 1

    cat_voc = {}
    cat_voc["default_cat"] = 0
    index = 1
    for key, value in sorted_cat_dict:
        cat_voc[key] = index
        index += 1

    cPickle.dump(uid_voc, open(user_vocab, "wb"))
    cPickle.dump(mid_voc, open(item_vocab, "wb"))
    cPickle.dump(cat_voc, open(cate_vocab, "wb"))