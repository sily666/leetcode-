[TOC]



# leetcode-DFS之回溯

回溯是 DFS 中的一种技巧。回溯法采用试错的思想，主要是在搜索尝试过程中寻找问题的解，当发现已不满足求解条件时，就“回溯”返回，尝试别的路径。

- 普通 DFS 主要用在 **可达性问题** ，这种问题只需要执行到特点的位置然后返回即可。
- 而 Backtracking 主要用于求解 **排列组合** 问题，例如有 { 'a','b','c' } 三个字符，求解所有由这三个字符排列得到的字符串，这种问题在执行到特定的位置返回之后还会继续执行求解过程。

回溯的本质就是暴力枚举所有可能。有时候可以通过剪枝去除一些根本不可能是答案的分支。

### 代码模板

```java
private void backtrack("原始参数") {
    //终止条件(递归必须要有终止条件)
    if ("终止条件") {
        //一些逻辑操作（可有可无，视情况而定）
        return;
    }

    for (int i = "for循环开始的参数"; i < "for循环结束的参数"; i++) {
        //一些逻辑操作（可有可无，视情况而定）

        //做出选择

        //递归
        backtrack("新的参数");
        //一些逻辑操作（可有可无，视情况而定）

        //撤销选择
    }
}
```

### 题型一：排列、组合、子集相关问题

#### [46. 全排列](https://leetcode-cn.com/problems/permutations/)

不含重复元素

```
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

visit数组：记录每条路径中使用过的数

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res = []
        visit = [0] * n
        def dfs(arr):
            if len(arr) == n:
                res.append(arr[:])
            for i in range(n):
                if visit[i]:
                    continue
                arr.append(nums[i])
                visit[i] = 1
                dfs(arr[:])
                visit[i] = 0
                arr.pop()
        dfs([])
        return res
```

#### [47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

包含重复数字

```
输入：nums = [1,1,2]
输出：[[1,1,2],[1,2,1],[2,1,1]]
```

**考虑重复元素一定要优先排序**，将重复的都放在一起，便于找到重复元素和剪枝

**剪枝条件1：**不能产生重复的排列。重复的排列是怎么产生的？

​		比如[1,1,2]，先选第一个1和先选第二个1，往后产生的排列是一样的。因此选第一个数时，不用考虑第二个		1。再比如，已选了2，现在，选第一个1，和选第二个1，往后产生的排列也是一样的。
​		它们都是 “同一层” 的选择出现重复，或者说，当前可选的选项出现重复。

**剪枝条件2：**一个数字不能重复地被选。比如[1,1,2]，第一个1只能在一个排列中出现一次。

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res = []
        visit = [0] * n
        nums.sort()
        def dfs(arr):
            if len(arr) == n:
                res.append(arr[:])
            for i in range(n):
                if visit[i]:
                    continue
                if i > 0 and nums[i] == nums[i-1] and not visit[i-1]:
                    continue
                arr.append(nums[i])
                visit[i] = 1
                dfs(arr[:])
                visit[i] = 0
                arr.pop()
        dfs([])
        return res
```

#### [剑指 Offer 38. 字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)

输入一个字符串，打印出该字符串中字符的所有排列。

```
输入：s = "abc"
输出：["abc","acb","bac","bca","cab","cba"]
```

```python
class Solution:
    def permutation(self, s: str) -> List[str]:
        c, res = list(s), []
        def dfs(x):
            if x == len(c) - 1:
                res.append(''.join(c))   # 添加排列方案
                return
            dic = set()
            for i in range(x, len(c)):
                if c[i] in dic: continue # 重复，因此剪枝
                dic.add(c[i])
                c[i], c[x] = c[x], c[i]  # 交换，将 c[i] 固定在第 x 位
                dfs(x + 1)               # 开启固定第 x + 1 位字符
                c[i], c[x] = c[x], c[i]  # 恢复交换
        dfs(0)
        return res
```



#### [39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)

给定一个**无重复元素**的数组 `candidates` 和一个目标数 `target` ，找出 `candidates` 中所有可以使数字和为 `target` 的组合。`candidates` 中的数字可以无限制重复被选取。

```
输入：candidates = [2,3,6,7], target = 7,
所求解集为：[[7],[2,2,3]]
```

（错误代码）

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        n = len(candidates)
        res = []
        def dfs(arr):
            if sum(arr) > target:
                return
            if sum(arr) == target:
                res.append(arr[:])
            for i in range(n):
                arr.append(candidates[i])
                dfs(arr[:])
                arr.pop()
        dfs([])
        return res
```

上述代码存在两个问题：

1.上述代码输出[[2,2,3],[2,3,2],[3,2,2],[7]]会出现重复值

2.每次都要求解sum值，较为繁琐

如何解决？

1.方法是调整下一次递归的值，只要数组序列号 >= 前一次所处的序列号的数组对应值

2.把sum值作为参数，每次只要加上push的数值

```js
var combinationSum = function(candidates, target) {
    let res = [];
    const fn = (arr, start, sum) => {
        if(sum > target) return;
        if(sum == target){
            res.push(arr.slice());
            return;
        }
        for(let l = start; l < candidates.length; l++){
            arr.push(candidates[l])
            fn(arr.slice(), l, sum+candidates[l])
            arr.pop()    
        }
    }
    fn([], 0, 0)
    return res;
};
```

#### [40. 组合总和 II](https://leetcode-cn.com/problems/combination-sum-ii/)

`candidates` 中的每个数字在每个组合中只能使用一次。

```
输入: candidates = [10,1,2,7,6,1,5], target = 8,
所求解集为:[[1, 7],[1, 2, 5],[2, 6],[1, 1, 6]]
```

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        n = len(candidates)
        res = []
        candidates.sort()
        def dfs(arr, start, sums):
            if sums > target: return
            if sums == target:
                res.append(arr[:])
                return
            for i in range(start, n):
                if i > start and candidates[i] == candidates[i-1]:
                    continue
                arr.append(candidates[i])
                dfs(arr, i + 1, sums + candidates[i]) #注意：这里是i+1,不是start+1
                arr.pop()
        dfs([], 0, 0)
        return res
```

或者用一个额外的set表来记录，删除重复。

```js
var combinationSum2 = function(candidates, target) {
    let res = [];
    let set = new Set();
    let dfs = (start,arr,result) => { //老套路了，dfs回溯模板代码
        if(result === target){ 
            let tmp = arr.slice().sort((a,b)=>a-b).join('.'); // arr.slice() => 拷贝数组 sort => 升序 join =>转化成字符串(因为Set.has 不能去重数组，所以要先转换成字符串))
            if(!set.has(tmp)){  // 判断有无重复路径
                res.push(arr.slice());
            }
            set.add(tmp); // 每遍历一次，就将该路径的字符串保存起来，方便查重
            return;
        }else if(result > target){ // 老套路代码了
            return;
        }
        for(let i = start;i<candidates.length;i++){ // 下面全是老套路代码 回溯经典写法
            arr.push(candidates[i]);
            dfs(i+1,arr,result+candidates[i]);
            arr.pop();
        }
    }
    dfs(0,[],0); 
    return res;
};
```

#### [216. 组合总和 III](https://leetcode-cn.com/problems/combination-sum-iii/)

找出所有相加之和为 ***n*** 的 **k** 个数的组合**。**组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。

```
输入: k = 3, n = 9
输出: [[1,2,6], [1,3,5], [2,3,4]]
```

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        res = []
        def dfs(arr, start, sums):
            if sums > n: return
            if len(arr) > k: return 
            if sums == n and len(arr) == k:
                res.append(arr[:])
                return
            for i in range(start, 10):
                arr.append(i)
                dfs(arr, i + 1, sums + i)
                arr.pop()
        dfs([], 1, 0)
        return res
```

#### [77. 组合](https://leetcode-cn.com/problems/combinations/)

给定两个整数 *n* 和 *k*，返回 1 ... *n* 中所有可能的 *k* 个数的组合。

```
输入: n = 4, k = 2
输出:[[2,4],[3,4],[2,3],[1,2],[1,3],[1,4]]
```

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        def dfs(arr, start):
            if len(arr) == k:
                res.append(arr[:])
            for i in range(start, n):
                arr.append(i+1)
                dfs(arr, i + 1)
                arr.pop()
        dfs([], 0)
        return res
```

#### [78. 子集](https://leetcode-cn.com/problems/subsets/)

给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的子集（幂集）。

```
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

**递归思想1**：单看每个元素，都有两种选择：选入子集中，或不选入子集中。

比如`[1,2,3]`，先考察 1，选或不选，不管选没选 1，都再考察 2，选或不选，以此类推。

```js
const subsets = (nums) => {
  const res = [];

  const dfs = (index, list) => {
    if (index == nums.length) { // 指针越界
      res.push(list.slice());   // 加入解集
      return;                   // 结束当前的递归
    }
    list.push(nums[index]); // 选择这个元素
    dfs(index + 1, list);   // 往下递归
    list.pop();             // 递归结束，撤销选择
    dfs(index + 1, list);   // 不选这个元素，往下递归
  };

  dfs(0, []);
  return res;
};

```

**递归思想2**：我们不设置递归的出口，但每次递归调用，传入的指针都基于当前选择+1，可选的范围变小了，即 for 循环的遍历范围变小了，一直递归到「没有可选的数字」，for 循环就不会落入递归，自然结束掉，整个DFS就结束了。

```js
const subsets = (nums) => {
  const res = [];

  const dfs = (index, list) => {
    res.push(list.slice());
    for (let i = index; i < nums.length; i++) {
      list.push(nums[i]);
      dfs(i + 1, list);
      list.pop();
    }
  };

  dfs(0, []);
  return res;
};
```

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res = []
        def dfs(arr, index):
            res.append(arr[:])
            for i in range(index, n):
                dfs(arr + [nums[i]], i + 1)
        dfs([], 0)
        return res
```

#### [90. 子集 II](https://leetcode-cn.com/problems/subsets-ii/)

给你一个整数数组 `nums` ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。

```
输入：nums = [1,2,2]
输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
```

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)
        nums.sort() #务必排序
        def dfs(arr, index):
            res.append(arr[:])
            for i in range(index, n):
                if nums[i] == nums[i-1] and i > index:
                    continue
                dfs(arr + [nums[i]], i + 1)
        dfs([], 0)
        return res
```

#### [60. 排列序列](https://leetcode-cn.com/problems/permutation-sequence/)

难度：hard

给出集合 `[1,2,3,...,n]`，其所有元素共有 `n!` 种排列。按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下："123"，"132"，"213"，"231"，"312"，"321"。给定 n 和 k，返回第 k 个排列。

```
输入：n = 3, k = 3
输出："213"
```

按照46题全排列的代码来解决，会超时。

```python
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        res = []
        visit = [0] * n
        def dfs(arr):
            if len(arr) == n:
                res.append(arr[:])
            for i in range(n):
                if visit[i]:
                    continue
                visit[i] = 1
                dfs(arr + str(i+1))
                visit[i] = 0
                if len(res) == k:
                    return res[-1]
        return dfs('')
```

**剪枝还不够充分，还可以优化**

每次都落入正确的分支，“空降”的感觉，这样就不用回溯找别的分支了，大大减少了不必要的搜索。如果完全没有剪枝，时间复杂度是`O(n!)`，充分的剪枝后，时间复杂度变为 `O(n^2)

比如: 首数字为1,后面有组成两个数123,132,可以组成2个数.当首数字为2,3同样都是。所以我们要找k = 3的数字 ,我们只需要 3/2 便可找到首数字什么,

```python
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        fac = 1
        nums = []
        for i in range(1, n + 1):
            nums.append(i)
            fac *= i
        k -= 1
        resStr = ''
        while len(nums) > 0:
            fac = fac // len(nums)
            index = math.floor(k / fac)
            resStr += str(nums[index])
            nums.pop(index)
            k = k % fac
        return resStr
```

### 题型二：Flood Fill

Flood 是「洪水」的意思，Flood Fill 直译是「泛洪填充」的意思，体现了洪水能够从一点开始，迅速填满当前位置附近的地势低的区域。

#### [733. 图像渲染](https://leetcode-cn.com/problems/flood-fill/)

将给定位置及其上下左右相同数字的点改为`newcolor`

```
输入: 
image = [[1,1,1],[1,1,0],[1,0,1]]
sr = 1, sc = 1, newColor = 2
输出: [[2,2,2],[2,2,0],[2,0,1]]
解析: 
在图像的正中间，(坐标(sr,sc)=(1,1)),
在路径上所有符合条件的像素点的颜色都被更改成2。
注意，右下角的像素没有更改为2，
因为它不是在上下左右四个方向上与初始点相连的像素点。
```

```python
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]: 
        row, col = len(image), len(image[0])
        p = image[sr][sc]
        if p == newColor:
            return image

        def dfs(x: int, y: int) -> None:
            if x in (-1, row) or y in (-1, col) or image[x][y] != p:
                return 
            image[x][y] = newColor

            for i in (-1,1):
                dfs(x+i, y)
                dfs(x, y+i)

            return image
        
        return dfs(sr, sc)
```



#### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

```
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
```

```python
class Solution:
    def numIslands(self, grid: [[str]]) -> int:
        def dfs(i, j):
            if not 0 <= i < len(grid) or not 0 <= j < len(grid[0]) or grid[i][j] == '0': 				return
            grid[i][j] = '0' #将遍历过的地改为0
            dfs(i + 1, j)
            dfs(i, j + 1)
            dfs(i - 1, j)
            dfs(i, j - 1)
            
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    dfs(i, j)
                    count += 1
        return count
```

#### [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

```
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
输出：6
```

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        res = 0

        def dfs(i, j):
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] == 0:
                return 0
            grid[i][j] = 0
            return 1 + dfs(i + 1, j) + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i, j - 1)

        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    res = max(res, dfs(i, j))
        return res
```

#### [130. 被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/)

给你一个 `m x n` 的矩阵 `board` ，由若干字符 `'X'` 和 `'O'` ，找到所有被 `'X'` 围绕的区域，并将这些区域里所有的 `'O'` 用 `'X'` 填充。

```
X X X X         X X X X
X O O X   -> 	X X X X
X X X X   		X X X X
X O O X			X O O X
```

从外围开始遍历，将与外围相连的‘O’改为‘A’，剩下的O就是需要改为‘X’的区域。

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        # 如果数组长或宽小于等于2，则不需要替换
        if len(board) <= 2 or len(board[0]) <= 2:
            return
        row, col = len(board), len(board[0])
        
        def dfs(i, j):
            if i < 0 or j < 0 or i >= row or j >= col or board[i][j] != 'O':
                return
            board[i][j] = 'A'
            
            dfs(i - 1, j)
            dfs(i + 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)
        
        # 从外围开始
        for i in range(row):
            dfs(i, 0)
            dfs(i, col-1)
        
        for j in range(col):
            dfs(0, j)
            dfs(row-1, j)
        
        # 最后完成替换
        for i in range(row):
            for j in range(col):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == 'A':
                    board[i][j] = 'O'
```

https://leetcode-cn.com/problems/surrounded-regions/solution/yuan-di-xiu-gai-de-dfs-python130-bei-wei-rao-de-qu/

#### [417. 太平洋大西洋水流问题](https://leetcode-cn.com/problems/pacific-atlantic-water-flow/)

给定一个 m x n 的非负整数矩阵来表示一片大陆上各个单元格的高度。“太平洋”处于大陆的左边界和上边界，而“大西洋”处于大陆的右边界和下边界。

规定水流只能按照上、下、左、右四个方向流动，且只能从高到低或者在同等高度上流动。

请找出那些水流既可以流动到“太平洋”，又能流动到“大西洋”的陆地单元的坐标。

```
给定下面的 5x5 矩阵:
  太平洋 ~   ~   ~   ~   ~ 
       ~  1   2   2   3  (5) *
       ~  3   2   3  (4) (4) *
       ~  2   4  (5)  3   1  *
       ~ (6) (7)  1   4   5  *
       ~ (5)  1   1   2   4  *
          *   *   *   *   * 大西洋
返回:
[[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (上图中带括号的单元).
```

```python
class Solution:
    def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:
        if not matrix or not matrix[0]: return []
        # 流向太平洋的位置
        res1 = set()
        # 流向大西洋的位置
        res2 = set()
        row = len(matrix)
        col = len(matrix[0])

        # 从边界遍历
        def dfs(i, j, res):
            res.add((i, j))
            for x, y in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                tmp_i = i + x
                tmp_j = j + y
                if 0 <= tmp_i < row and 0 <= tmp_j < col and matrix[i][j] <= matrix[tmp_i][tmp_j] and (tmp_i, tmp_j) not in res: 
                    dfs(tmp_i, tmp_j, res)
        # 太平洋
        for i in range(row):
            dfs(i, 0, res1)
        # 太平洋
        for j in range(col):
            dfs(0, j, res1)
        # 大西洋
        for i in range(row):
            dfs(i, col - 1, res2)
        # 大西洋
        for j in range(col):
            dfs(row - 1, j, res2)

        return list(res1 & res2)
```

#### [79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word` 。如果 `word` 存在于网格中，返回 `true` ；否则，返回 `false` 。

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
```

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m, n = len(board), len(board[0])
        used = [[False] * n for _ in range(m) ]
        def dfs(i, j, index):
            if index == len(word): return True #成功条件写在前
            if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[index] or used[i][j]:
                return False
            used[i][j] = True
            if dfs(i-1, j, index+1) or dfs(i+1, j, index+1) or dfs(i, j-1, index+1) or dfs(i, j+1, index+1):
                return True
            used[i][j] = False
            return False
        
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0] and dfs(i, j, 0):
                    return True
        return False
```

https://leetcode-cn.com/problems/word-search/solution/shou-hua-tu-jie-79-dan-ci-sou-suo-dfs-si-lu-de-cha/

### 题型三：字符串中的回溯问题

#### [17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。数字映射的字母与电话按键相同：`['abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']`

```
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        n = len(digits)
        res = []
        arr = ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
        if not digits: return []
        def dfs(resStr, index):
            if len(resStr) == n:
                res.append(resStr)
                return
            s = arr[int(digits[index]) - 2]
            for i in s:
                dfs(resStr + i, index + 1)
        dfs('', 0)
        return res
```

#### [784. 字母大小写全排列](https://leetcode-cn.com/problems/letter-case-permutation/)

给定一个字符串`S`，通过将字符串`S`中的每个字母转变大小写，我们可以获得一个新的字符串。返回所有可能得到的字符串集合。

```
输入：S = "a1b2"
输出：["a1b2", "a1B2", "A1b2", "A1B2"]
```

```python
class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        n = len(s)
        res = []
        def dfs(resStr, index):
            if len(resStr) == n:
                res.append(resStr)
                return
            if 'a' <= s[index] <= 'z' or 'A' <= s[index] <= 'Z':
                dfs(resStr + s[index].swapcase(), index + 1)
            dfs(resStr + s[index], index + 1)
        dfs('', 0)
        return res
```

#### [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

```
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]
```

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        def dfs(resStr, left, right):
            if left < right or left > n: # 剪枝条件在前，终止条件在后
                return
            if len(resStr) == 2 * n:
                res.append(resStr)
                return 
            dfs(resStr + '(', left + 1, right)
            dfs(resStr + ')', left, right + 1)
        dfs('', 0, 0)
        return res
```

#### [131. 分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/)

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是 **回文串** 。返回 `s` 所有可能的分割方案。

```
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
```

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        def dfs(arr, start):
            if start == len(s):
                res.append(arr[:])
                return
            for i in range(1, len(s) - start + 1):
                sub = s[start: start + i]
                if sub == sub[::-1]:
                    dfs(arr + [sub], start + i)
        dfs([], 0)
        return res
```

#### [93. 复原 IP 地址](https://leetcode-cn.com/problems/restore-ip-addresses/)

给定一个只包含数字的字符串，用以表示一个 IP 地址，返回所有可能从 s 获得的 有效 IP 地址 。你可以按任何顺序返回答案。

有效 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。

```
输入：s = "25525511135"
输出：["255.255.11.135","255.255.111.35"]
```

剪枝条件：

1. 一个片段的长度是 1~3
2. 片段的值范围是 0~255
3. 不能是 "0x"、"0xx" 形式

结束条件：

- 目标是生成 4 个有效片段，并且要耗尽 IP 的字符。
- 如果满4个有效片段，但没耗尽字符，不是想要的解，不继续往下递归，提前回溯。

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def dfs(subRes, index):
            if len(subRes) == 4 and index < n: return
            if len(subRes) == 4 and index == n:
                res.append('.'.join(subRes))
                
            for i in range(1, 4):
                if index + i - 1 > n: return
                if i != 1 and s[index] == '0': return
                sub = s[index: index + i]
                if i == 3 and int(sub) > 255: return
                subRes.append(sub)
                dfs(subRes, index + i)
                subRes.pop()
        res = []
        n = len(s)
        dfs([], 0)
        return res
```

### 题型四：游戏问题

#### [51. N 皇后](https://leetcode-cn.com/problems/n-queens/)

n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。任何两个皇后都不能处于同一条横行、纵行或斜线上。

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        board = [['.'] * n for _ in range(n)]
        res = []
        def isValid(row, col):
            for i in range(row):
                for j in range(n):
                    if (board[i][j] == 'Q') and (j == col or i+j == row + col or i-j == row - col): #发现了皇后，并且和自己同列/对角线
                        return False
            return True

        def dfs(row):
            if row == n:
                strBoard = board[:]
                for i in range(n):
                    strBoard[i] = ''.join(strBoard[i])
                res.append(strBoard)
                return
            for i in range(n):
                if isValid(row, i):
                    board[row][i] = 'Q'
                    dfs(row + 1)
                    board[row][i] = '.'
        dfs(0)
        return res
```

用空间换时间，进行优化
这道题需要记录过去的选择，即皇后放置的位置，才能结合约束条件去做剪枝。

最好是用三个数组或 HashSet 去记录出现过皇后的列们、正对角线们、反对角线们，用空间换取时间。

优化后的代码：基于集合的回溯

```js
const solveNQueens = (n) => {
  const board = new Array(n);
  for (let i = 0; i < n; i++) {
    board[i] = new Array(n).fill('.');
  }

  const cols = new Set();  // 列集，记录出现过皇后的列
  const diag1 = new Set(); // 正对角线集
  const diag2 = new Set(); // 反对角线集
  const res = [];

  const helper = (row) => {
    if (row == n) {
      const stringsBoard = board.slice();
      for (let i = 0; i < n; i++) {
        stringsBoard[i] = stringsBoard[i].join('');
      }
      res.push(stringsBoard);
      return;
    }
    for (let col = 0; col < n; col++) {
      // 如果当前点的行列对角线都没有皇后，即可选择，否则，跳过
      if (!cols.has(col) && !diag1.has(row + col) && !diag2.has(row - col)) { 
        board[row][col] = 'Q';  // 放置皇后
        cols.add(col);          // 记录放了皇后的列
        diag1.add(row + col);   // 记录放了皇后的正对角线
        diag2.add(row - col);
        helper(row + 1);
        board[row][col] = '.';  // 撤销该点的皇后
        cols.delete(col);       // 对应的记录也删一下
        diag1.delete(row + col);
        diag2.delete(row - col);
      }
    }
  };
  helper(0);
  return res;
};
```

#### [37. 解数独](https://leetcode-cn.com/problems/sudoku-solver/)

数独规则如下：数字1-9在每行每列每个以粗实线分割的3x3格内只能出现一次。

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        def hasConflit(r, c, val):
            for i in range(9):
                if board[i][c] == val or board[r][i] == val: #每行每列有重复数字
                    return True
            subRow = math.floor(r / 3) * 3 #3x3格起始下标0,3,6
            subCol = math.floor(c / 3) * 3
            for i in range(3):
                for j in range(3):
                    if val == board[subRow + i][subCol + j]: #3x3格内有重复
                        return True
            return False
        
        def fill(i, j):
            if j == 9: #一行遍历完毕 继续向下一行遍历
                i += 1
                j = 0
                if i == 9: # 遍历到（9,9）时，遍历成功
                    return True
            if board[i][j] != '.': #如果存在数字，向下一格遍历
                return fill(i, j + 1)
            for num in range(1, 10): #依次遍历每个数字
                if hasConflit(i, j, str(num)): continue
                board[i][j] = str(num)
                if fill(i, j + 1): #如果基于它，填下一格，最后可以解出数独，直接返回true
                    return True
                board[i][j] = '.' #如果基于它，填下一格，填1-9都不行，回溯，恢复为空白格
            return False #尝试了1-9，每个都往下递归，都不能做完，返回false
        
        fill(0, 0)
```

优化后的代码  其实可以用三个数组：

rows 数组，长度为 9，对应每一行都有一个 hashSet，hashSet 存可填的数字。
cols 数组，长度为 9，对应每一列都有一个 hashSet，hashSet 存可填的数字。
blocks 数组，长度为 9，对应 9 个框都有一个 hashSet，hashSet 存可填的数字。

如果当前格子，选填的数，在对应的三个 hashSet 里，有一个里面没有它，那就不能选。这样就避免每次去遍历判断，用空间换取时间。

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None: 
        options = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        rows = [set(options) for _ in range(9)]
        cols = [set(options) for _ in range(9)]
        blocks = [set(options) for _ in range(9)]
        
        def getBlockIndex(i, j):
        	return i // 3 * 3 + j //3
        
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    rows[i].remove(board[i][j])
                    cols[j].remove(board[i][j])
                    blocks[getBlockIndex(i, j)].remove(board[i][j])
        def fill(i, j):
            if j == 9:
                i += 1
                j = 0
                if i == 9:
                    return True
            if board[i][j] != '.': return fill(i, j + 1)
            for num in range(1, 10):
                s = str(num)
                if s not in rows[i] or s not in cols[j] or s not in blocks[getBlockIndex(i, j)]:
                    continue
                board[i][j] = s
                rows[i].remove(s)
                cols[j].remove(s)
                blocks[getBlockIndex(i, j)].remove(s)
                if fill(i, j + 1): return True
                board[i][j] = '.'
                rows[i].add(s)
                cols[j].add(s)
                blocks[getBlockIndex(i, j)].add(s)
            return False
        fill(0, 0)

```

https://leetcode-cn.com/problems/sudoku-solver/solution/shou-hua-tu-jie-jie-shu-du-hui-su-suan-fa-sudoku-s/

#### [488. 祖玛游戏](https://leetcode-cn.com/problems/zuma-game/)

#### [529. 扫雷游戏](https://leetcode-cn.com/problems/minesweeper/)





