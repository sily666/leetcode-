# leetcode-BFS



可以求解最短路径等 **最优解** 问题：第一次遍历到目的节点，其所经过的路径为最短路径。应该注意的是，使用 BFS 只能求解无权图的最短路径，无权图是指从一个节点到另一个节点的代价都记为 1。

在程序实现 BFS 时需要考虑以下问题：

- 队列：用来存储每一轮遍历得到的节点；
- 标记：对于遍历过的节点，应该将它标记，防止重复遍历。

#### [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        res = 0
        
        def bfs(i, j):
            queue = [(i, j)]
            grid[i][j] = 0
            area = 1
            while queue:
                x, y = queue.pop(0)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < m and 0 <= ny < n and grid[nx][ny]:
                        grid[nx][ny] = 0
                        area += 1
                        queue.append((nx, ny))
            return area

        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    res = max(res, bfs(i, j))
        return res
```

#### [1091. 二进制矩阵中的最短路径](https://leetcode-cn.com/problems/shortest-path-in-binary-matrix/)

给你一个 n x n 的二进制矩阵 grid 中，返回矩阵中最短 畅通路径 的长度。如果不存在这样的路径，返回 -1 。

二进制矩阵中的 畅通路径 是一条从 左上角 单元格（即，(0, 0)）到 右下角 单元格（即，(n - 1, n - 1)）的路径，该路径同时满足下述要求：

路径途经的所有单元格都的值都是 0 。路径中所有相邻的单元格应当在 8 个方向之一 上连通（即，相邻两单元之间彼此不同且共享一条边或者一个角）。

畅通路径的长度 是该路径途经的单元格总数。

```
输入：grid = [[0,1],[1,0]]
输出：2
```

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if not grid or grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1
        elif n <= 2:
            return n
        queue = [(0, 0, 1)]
        grid[0][0] = 1
        while queue:
            i, j, step = queue.pop(0)           
            for dx, dy in [(-1,-1), (1,0), (0,1), (-1,0), (0,-1), (1,1), (1,-1), (-1,1)]:
                if i+dx == n-1 and j+dy == n-1:
                    return step + 1
                if 0 <= i+dx < n and 0 <= j+dy < n and grid[i+dx][j+dy] == 0:
                    queue.append((i+dx, j+dy, step+1))
                    grid[i+dx][j+dy] = 1  # mark as visited                   
        return -1

```

#### [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。给你一个整数 n ，返回和为 n 的完全平方数的 最少数量 。

```
输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4
```

```python
class Solution:
    def numSquares(self, n: int) -> int:
        res = 0
        queue = [(n, res)]
        while queue:
            num, j = queue.pop(0)
            for i in range(int(math.sqrt(num)), 0, -1):
                if num - i * i == 0: return j + 1
                queue.append((num - i * i, j + 1))
```

#### [127. 单词接龙](https://leetcode-cn.com/problems/word-ladder/)

字典 `wordList` 中从单词 `beginWord` 和 `endWord` 的 **转换序列** 是一个按下述规格形成的序列：序列中第一个单词是` beginWord` 。

+ 序列中最后一个单词是` endWord `。
+ 每次转换只能改变一个字母。
+ 转换过程中的中间单词必须是字典 `wordList` 中的单词。

给你两个单词 `beginWord`和 `endWord` 和一个字典 `wordList `，找到从`beginWord `到 `endWord` 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0

```
输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
输出：5
解释：一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog", 返回它的长度 5。
```

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordSet = set(wordList)
        queue = [(beginWord, 1)]
        while queue:
            s, slen = queue.pop(0)
            if s == endWord: return slen
            for i in range(len(s)):
                for c in range(97, 123):
                    newword = s[0:i] + chr(c) + s[i+1: len(s)]
                    if newword in wordSet:
                        queue.append((newword, slen + 1))
                        wordSet.remove(newword)
        return 0
```

