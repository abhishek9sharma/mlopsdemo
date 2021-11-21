from node2vec import Node2Vec
from gensim.models import Word2Vec
import networkx as nx


def fit_node2vec(X_train_graph, features, target, param_dict=None):
    """Trains a node2vec model given information about users and items

    :param X_train_graph: a pandas dataframe contain click interactions between users and items
    :param features: features which constitute the context between user and item nodes
    :param target: a string value which corresponds to the merchant nodes
    :param param_dict: parameter settings for the node2vec model training
    """

    X_train_graph["user_node"] = X_train_graph.user_id.apply(lambda x: "user_" + str(x))
    X_train_graph["merchant_node"] = X_train_graph[target[0]].apply(
        lambda x: "merchant_" + str(x)
    )
    # short data as of now
    # X_train_graph_small = X_train_graph.iloc[0:10]
    G = nx.from_pandas_edgelist(
        X_train_graph,
        source="user_node",
        target="merchant_node",
        edge_attr=features,
        create_using=nx.DiGraph(),
    )

    if not param_dict:
        param_dict = {
            "workers": 2,
            "dimensions": 10,
            "walk_length": 10,
            "num_walks": 20,
            "window": 2,
            "min_count": 1,
            "batch_words": 4,
        }

    node2vec = Node2Vec(
        G,
        workers=param_dict["workers"],
        dimensions=param_dict["dimensions"],
        walk_length=param_dict["walk_length"],
        num_walks=param_dict["num_walks"],
    )

    model = node2vec.fit(
        window=param_dict["window"],
        min_count=param_dict["min_count"],
        batch_words=param_dict["batch_words"],
    )
    return model


def get_ordered_preds_n2v(user_id, merchants, n2vmodel, return_scores=False):
    """This method reaturns a list of merchants/dictionary of merchants and scores
    that are most similar to a user vector in the learned node2vec model

    :param user_id: id of the user
    :param merchants: a dictionary of key value pairs where key is the
            merchant_id and value is the trained vector
    :param n2vmodel: trained node2vec model
    :param return_scores: a boolean flag which specifies if a list of merchants needs
            to be returned or a dictionary of merchants with the similarty scores
    :return: a list or dictionary of next merchants which should be recommended
    """

    # add condition for new user
    num_merchants = len(merchants)
    user_id_key = "user_" + str(user_id)
    if user_id_key not in n2vmodel.wv.key_to_index.keys():
        return [0] * num_merchants

    user_to_merchant_sim = {}
    for merchant in merchants:
        sim = n2vmodel.wv.similarity(user_id_key, merchant)
        mid = merchant.replace("merchant_", "")
        user_to_merchant_sim[mid] = sim
    user_to_merchant_sim = dict(
        sorted(
            user_to_merchant_sim.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    if return_scores:
        return user_to_merchant_sim
    else:
        return list(user_to_merchant_sim.keys())


def get_merchant_vecs(n2vmodel):
    """Extract only the merchant related key values
        pairs from node2vec model keys

    :param n2vmodel: a node2vec model object
    :return: a dictionary of key value pairs where key is the
            merchant_id and value is the trained vector
    """

    all_nodes = n2vmodel.wv.key_to_index.keys()
    merchants = [n for n in all_nodes if "merchant" in n]
    item_dict = {merchant: n2vmodel.wv[merchant] for merchant in merchants}
    return item_dict
