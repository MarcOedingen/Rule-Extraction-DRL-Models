from ruleex import deepred

def reshape_weights(weights):
    """
    reshapes the layers of the loaded weights in useful form for the rule-extraction algorithm (DeepRED).
    :param weights:
    :return:
    """
    resh_weights = [[], []]
    for i in range(len(weights)):
        if i % 2 == 0:
            resh_weights[0].append(weights[i])
        else:
            resh_weights[1].append(weights[i])
    return resh_weights


def get_layer_sizes(weights):
    """
    computes a layer sizes of NN from weights
    :param weights: weights of the NN (weights[0] is a list of connection weights and weights[1] is a list of biases)
    :return: a list of the layer sizes
    """
    layer_sizes = [len(weights[0][0])]
    for b in weights[1]:
        layer_sizes.append(len(b))
    return layer_sizes


def Ruletree_to_string(rule, depth=0):
    """
    Converts the given Rule-Tree to runnable if-else statements by recursion in python code. Given a rule, we start
    with the root of the Rule-Tree and depth = 0 and traverse the tree recursively.
    :param rule:
    :param depth:
    :return:
    """
    indent = "  " * depth
    if isinstance(rule, deepred.AxisRule):
        print(f"{indent} if obs[{rule.i}] <= {round(rule.b, 2)}:")
        Ruletree_to_string(rule.false_branch, depth + 1)
        print(f"{indent} else:")
        Ruletree_to_string(rule.true_branch, depth + 1)
    else:
        iterator = iter(rule.class_set)
        print(f"{indent} return {next(iterator, None)}")
