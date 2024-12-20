import numpy as np
import torch

MAX_STACK = 1024
MAX_FULL_DEPTH = 10


class NType:
    """
    The enumeration class for GP node types.
    """

    VAR = 0  # variable
    CONST = 1  # constant
    UFUNC = 2  # unary function
    BFUNC = 3  # binary function
    TFUNC = 4  # ternary function
    TYPE_MASK = 0x7F  # node type mask
    OUT_NODE = 1 << 7  # out node flag
    UFUNC_OUT = UFUNC + OUT_NODE  # unary function, output node
    BFUNC_OUT = BFUNC + OUT_NODE  # binary function, output node
    TFUNC_OUT = TFUNC + OUT_NODE  # ternary function, output node


class Func:
    """
    The enumeration class for GP function types.
    """

    IF = 0

    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    POW = 5
    MAX = 6
    MIN = 7
    LT = 8
    GT = 9
    LE = 10
    GE = 11

    SIN = 12
    COS = 13
    TAN = 14
    SINH = 15
    COSH = 16
    TANH = 17
    LOG = 18
    EXP = 19
    INV = 20
    NEG = 21
    ABS = 22
    SQRT = 23
    END = 24


FUNCS = [
    Func.IF,
    Func.ADD,
    Func.SUB,
    Func.MUL,
    Func.DIV,
    Func.POW,
    Func.MAX,
    Func.MIN,
    Func.LT,
    Func.GT,
    Func.LE,
    Func.GE,
    Func.SIN,
    Func.COS,
    Func.TAN,
    Func.SINH,
    Func.COSH,
    Func.TANH,
    Func.LOG,
    Func.EXP,
    Func.INV,
    Func.NEG,
    Func.ABS,
    Func.SQRT,
]

FUNCS_NAMES = [
    "if",  # 0
    "+",  # 1
    "-",  # 2
    "*",  # 3
    "/",  # 4
    "pow",  # 5
    "max",  # 6
    "min",  # 7
    "<",  # 8
    ">",  # 9
    "<=",  # 10
    ">=",  # 11
    "sin",  # 12
    "cos",  # 13
    "tan",  # 14
    "sinh",  # 15
    "cosh",  # 16
    "tanh",  # 17
    "log",  # 18
    "exp",  # 19
    "inv",  # 20
    "neg",  # 21
    "abs",  # 22
    "sqrt",  # 23
]


def dict2prob(prob_dict):
    # Probability Dictionary to Distribution Function
    assert len(prob_dict) > 0, "Empty probability dictionary"

    prob = np.zeros(len(FUNCS))

    for key, val in prob_dict.items():
        assert (
            key in FUNCS_NAMES
        ), f"Unknown function name: {key}, total functions are {FUNCS_NAMES}"
        idx = FUNCS_NAMES.index(key)
        prob[idx] = val

    # normalize
    prob = prob / prob.sum()

    return prob


def dict2cdf(prob_dict):
    # Probability Dictionary to Cumulative Distribution Function
    prob = dict2prob(prob_dict)

    return np.cumsum(prob)


def to_numpy(li):
    for idx, e in enumerate(li):
        if type(e) == torch.Tensor:
            li[idx] = e.cpu().numpy()
    return li


def dict2prob(prob_dict):
    # Probability Dictionary to Distribution Function
    assert len(prob_dict) > 0, "Empty probability dictionary"

    prob = np.zeros(len(FUNCS))

    for key, val in prob_dict.items():
        assert (
            key in FUNCS_NAMES
        ), f"Unknown function name: {key}, total functions are {FUNCS_NAMES}"
        idx = FUNCS_NAMES.index(key)
        prob[idx] = val

    # normalize
    prob = prob / prob.sum()

    return prob


def dict2cdf(prob_dict):
    # Probability Dictionary to Cumulative Distribution Function
    prob = dict2prob(prob_dict)

    return np.cumsum(prob)


def check_tensor(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device="cuda", requires_grad=False)
    else:
        x = x.to("cuda").detach().requires_grad_(False)
        return x


def str_tree(value, node_type, subtree_size):
    res = ""
    for i in range(0, subtree_size[0]):
        if (
            (node_type[i] == NType.UFUNC)
            or (node_type[i] == NType.BFUNC)
            or (node_type[i] == NType.TFUNC)
        ):
            res = res + FUNCS_NAMES[int(value[i])]
        elif node_type[i] == NType.VAR:
            res = res + f"x[{int(value[i])}]"
        elif node_type[i] == NType.CONST:
            res = res + f"{value[i]:.2f}"
        res += " "

    return res
