import logging 
import pandas as pd
import _pickle as cPickle
import random

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
    
def create_item2cate(instance_file):
    logger.info("creating item2cate dict")
    global item2cate
    instance_df = pd.read_csv(
        instance_file,
        sep="\t",
        names=["label", "user_id", "item_id", "timestamp", "cate_id"],
    )
    item2cate = instance_df.set_index("item_id")["cate_id"].to_dict() 
    
def get_sampled_data(instance_file, sample_rate):
    logger.info("getting sampled data...")
    global item2cate
    output_file = str(instance_file) + "_" + str(sample_rate)
    columns = ["label", "user_id", "item_id", "timestamp", "cate_id"]
    ns_df = pd.read_csv(instance_file, sep="\t", names=columns)
    items_num = ns_df["item_id"].nunique()
    items_with_popular = list(ns_df["item_id"])
    items_sample, count = set(), 0
    while count < int(items_num * sample_rate):
        random_item = random.choice(items_with_popular)
        if random_item not in items_sample:
            items_sample.add(random_item)
            count += 1
    ns_df_sample = ns_df[ns_df["item_id"].isin(items_sample)]
    ns_df_sample.to_csv(output_file, sep="\t", index=None, header=None)
    return output_file\

def negative_sampling_offline(
    instance_input_file, valid_file, test_file, valid_neg_nums=4, test_neg_nums=49
):

    columns = ["label", "user_id", "item_id", "timestamp", "cate_id"]
    ns_df = pd.read_csv(instance_input_file, sep="\t", names=columns)
    items_with_popular = list(ns_df["item_id"])

    global item2cate

    # valid negative sampling
    logger.info("start valid negative sampling")
    with open(valid_file, "r") as f:
        valid_lines = f.readlines()
    write_valid = open(valid_file, "w")
    for line in valid_lines:
        write_valid.write(line)
        words = line.strip().split("\t")
        positive_item = words[2]
        count = 0
        neg_items = set()
        while count < valid_neg_nums:
            neg_item = random.choice(items_with_popular)
            if neg_item == positive_item or neg_item in neg_items:
                continue
            count += 1
            neg_items.add(neg_item)
            words[0] = "0"
            words[2] = str(neg_item)
            words[3] = str(item2cate[neg_item])
            write_valid.write("\t".join(words) + "\n")