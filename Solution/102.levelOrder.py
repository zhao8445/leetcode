from typing import List


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        """二叉树的层序遍历"""
        if not root:
            return []
        queue = [root]
        res = [[root.val]]
        while queue:
            # node = queue.pop(0)
            tmp = []
            """需要将加入queue中的二叉树节点都弹尽遍历完毕，加上for循环"""
            for _ in range(len(queue)):
                node = queue.pop(0)
                if node.left:
                    queue.append(node.left)
                    tmp.append(node.left.val)
                if node.right:
                    queue.append(node.right)
                    tmp.append(node.right.val)
            if tmp:
                res.append(tmp)
        return res