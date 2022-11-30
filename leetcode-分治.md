# leetcode-分治

**分治算法三步走**：

1. 分解：按运算符分成左右两部分，分别求解
2. 解决：实现一个递归函数，输入算式，返回算式解
3. 合并：根据运算符合并左右两部分的解，得出最终解

#### [241. 为运算表达式设计优先级](https://leetcode-cn.com/problems/different-ways-to-add-parentheses/)

```
输入: "2*3-4*5"
输出: [-34, -14, -10, -10, 10]
解释: 
(2*(3-(4*5))) = -34   ((2*3)-(4*5)) = -14   ((2*(3-4))*5) = -10
(2*((3-4)*5)) = -10   (((2*3)-4)*5) = 10
```

```python
class Solution:
    def diffWaysToCompute(self, expression: str) -> List[int]:
        if expression.isdigit(): return [int(expression)]
        res = []
        for i, ch in enumerate(expression):
            if ch in ['+', '-', '*']: # 分解
                left = self.diffWaysToCompute(expression[:i])
                right = self.diffWaysToCompute(expression[i + 1:])
                # 合并
                for l in left:
                    for r in right:
                        if ch == '+':
                            res.append(l + r)
                        elif ch == '-':
                            res.append(l - r)
                        else:
                            res.append(l * r)
        return res
```



#### [95. 不同的二叉搜索树 II](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/)

给你一个整数 `n` ，请你生成并返回所有由 `n` 个节点组成且节点值从 `1` 到 `n` 互不相同的不同 **二叉搜索树** 。可以按 **任意顺序** 返回答案。

![95](./img/95.jpg)

```
输入：n = 3
输出：[[1,null,2,null,3],[1,null,3,2],[2,1,3],[3,1,null,null,2],[3,2,null,1]]
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        def buildTree(start, end):
            res = []
            if start > end: return [None]
            for i in range(start, end + 1):
                left = buildTree(start, i - 1)
                right = buildTree(i + 1, end)
                for l in left:
                    for r in right:
                        node = TreeNode(i)
                        node.left = l
                        node.right = r
                        res.append(node)
            return res
        if n == 0: return []
        return buildTree(1, n)
```

