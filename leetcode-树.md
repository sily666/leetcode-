# leetcode-树

#### [993. 二叉树的堂兄弟节点](https://leetcode-cn.com/problems/cousins-in-binary-tree/)

如果二叉树的两个节点深度相同，但 **父节点不同** ，则它们是一对*堂兄弟节点*。

给出了具有唯一值的二叉树的根节点 `root` ，以及树中两个不同节点的值 `x` 和 `y` 。是堂兄弟节点时，返回 `true` 。否则，返回 `false`。

深度遍历：存储每个节点的深度和父节点

```python
class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        depth = {}
        father = {}
        
        def dfs(root, parent):
            if not root:
                return
            depth[root.val] = depth[parent.val] + 1 if parent else 0
            father[root.val] = parent
            dfs(root.left, root)
            dfs(root.right, root)
        
        dfs(root, None)
        return depth[x] == depth[y] and father[x] != father[y]
```

广度遍历：按层遍历节点，存储父节点

```python
class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        if not root:
            return False

        queue = [root]
        father = {}
        while queue:
            temp = set()
            for _ in range(len(queue)):
                node, parent = queue.pop(0)
                if node.left:
                    queue.append((node.left, node))
                if node.right:
                    queue.append((node.right, node))
                temp.add(node.val)
                if node.val == x:
                    father[x] = parent
                if node.val == y:
                    father[y] = parent
                if x in temp and y in temp:
                    return father[x] != father[y]
        return False
```

