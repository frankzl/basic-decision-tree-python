from collections import Counter
from treelib import Node, Tree
# import pptree

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

CONST_PRECISION = 2

class DecisionTree:

    NODE_WIDTH = 20

    def __init__(self, maxdepth, X, y, column_names):
        self.maxdepth = maxdepth
        self.root = DtNode(X, y, 0, maxdepth, column_names)
    
    def draw(self):
        tree = Tree()
        tree = self.create_drawn_tree(tree, self.root, 1)
        tree.show()

    def create_drawn_tree(self, tree, node, node_id, parent_id=None):
        if parent_id is None:
            tree.create_node(node.info(), node_id)
        else:
            tree.create_node(node.info(), node_id, parent_id)
        if node.true_node:
            self.create_drawn_tree(tree, node.true_node, node_id*10+0, node_id)
        if node.true_node:
            self.create_drawn_tree(tree, node.false_node, node_id*10+1, node_id)
        return tree

    def get_distribution(self):
        return self.root.get_distribution()

    def get_purity(self, metric="gini"):
        return self.root.get_purity(metric)

    def create_split(self, feature, value):
        return self.root.create_split(feature, value)

    def auto_split(self):
        self.root.auto_split()

    def predict(self, data_point):
        return self.root.predict(data_point)

class DtNode:
    def __init__(self, X, y, depth, maxdepth, column_names, flag = None):
        self.split_val = None
        self.depth = depth
        self.maxdepth = maxdepth
        self.X = X
        self.y = y
        self.true_node = None
        self.false_node = None
        self.split_val = None
        self.column_names = column_names
        self.flag = flag

    def get_purity(self, metric="gini"):
        if metric=="gini":
            return round(self.get_gini(), CONST_PRECISION )
        else:
            return round(self.get_misclass(),CONST_PRECISION)

    def get_size(self):
        return len(self.y)

    def get_misclass(self):    
        purity = 1
        max_val = 0
        for key, value in self.get_distribution().items():
            max_val = max_val if value < max_val else value
        return purity - max_val

    def get_gini(self):    
        purity = 1
        for key, value in self.get_distribution().items():
            purity -= value**2
        return purity

    def auto_split(self):
        if self.get_purity() == 0:
            return
        if self.depth < self.maxdepth:
            feat, value = self.get_best_split()
            self.split_val = value
            self.split_feat = feat
            self.true_node, self.false_node = self.create_split(feat, value)
            self.true_node.auto_split()
            self.false_node.auto_split()

    def get_best_split(self, metric="gini"):
        features = len(self.X[0])
        best_splits = []
        for feat in range(features):
            improvements = self.get_all_split_improvements( feat, metric )
            best_splits.append( max( improvements, key=lambda x: x[1]))
        bestFeat = 0
        bestVal = 0
        for feat in range(features):
            if bestVal < best_splits[feat][1]:
                bestVal = best_splits[feat][0]
                bestFeat = feat
        return (bestFeat, bestVal)

    def get_all_split_improvements(self, feat, metric="gini"):
        unique_data = [list(x) for x in set(tuple(x) for x in self.X)]
        improvements = [ [x[feat], self.get_split_improvement(feat, x[feat], metric)] for x in unique_data ]
        unique_improvements = [list(x) for x in set(tuple(x) for x in improvements)]
        return unique_improvements 

    def get_split_improvement(self, feat, val, metric="gini"):
        return round(self.get_purity() - self.get_split_purity(feat, val, metric),CONST_PRECISION)

    def get_split_purity(self, feat, val, metric="gini"):
        true_node, false_node = self.create_split(feat, val)

        true_ratio  = true_node.get_size()/self.get_size()
        false_ratio = false_node.get_size()/self.get_size()

        return round(true_ratio * true_node.get_purity(metric)
                + false_ratio * false_node.get_purity(metric), CONST_PRECISION)

    def create_split(self, feat, val):
        self.split_val = val
        true_data = []
        true_y = []
        false_data = []
        false_y = []
        for idx, x in enumerate(self.X):
            if( x[feat] <= val):
                true_data.append(x) 
                true_y.append(self.y[idx])
            else:
                false_data.append(x)
                false_y.append(self.y[idx])

        true_node = DtNode(true_data, true_y, self.depth + 1, self.maxdepth, self.column_names, True)
        false_node = DtNode(false_data, false_y, self.depth + 1, self.maxdepth, self.column_names, False)

        return (true_node, false_node)

    def predict(self, data_point):
        if self.split_val is None:
            distr = self.get_distribution()
            maxkey = (max(distr, key=lambda k: distr[k]))
            return maxkey
        else:
            if data_point[self.split_feat] <= self.split_val:
                return self.true_node.predict(data_point)
            else:
                return self.false_node.predict(data_point)

    def info(self):
        info = ""

        if self.flag is not None:
            info += color.BOLD + color.RED + str(self.flag) + color.END + " | "

        if self.split_val is not None:
            info += color.BOLD + "IF "+str(self.column_names[self.split_feat]) + " <= " + str(self.split_val) + color.END + " | \t"
        # distr = str(dict(Counter(self.y)))
        # info += "Set: " + distr.ljust(20) + ""
        info += "distr: " + str(self.get_distribution()).ljust(20) + ""
        info += " gini: " + str(self.get_purity())
        return info + "\n"

    def get_distribution(self):
        c = Counter(self.y)
        distr = {}
        for idx in c:
            distr[idx] = round(c[idx]/len(self.y), CONST_PRECISION)
        return distr
