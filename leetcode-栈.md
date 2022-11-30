# leetcode-栈

[TOC]



#### [232. 用栈实现队列](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

```python
class MyQueue:
    def __init__(self):
        self.a = []
        self.b = []

    def push(self, x: int) -> None:
        while self.b:
            self.a.append(self.b.pop())
        self.a.append(x)
        while self.a:
            self.b.append(self.a.pop())

    def pop(self) -> int:
        return self.b.pop()

    def peek(self) -> int: # 返回队列开头的元素
        return self.b[-1]

    def empty(self) -> bool:
        return len(self.b) == 0

```

#### [225. 用队列实现栈](https://leetcode-cn.com/problems/implement-stack-using-queues/)

```python
class MyStack:
    def __init__(self):
        self.q = []

    def push(self, x: int) -> None:
        self.q.append(x)
        n = len(self.q)
        while n > 1:
            self.q.append(self.q.pop(0))
            n -= 1

    def pop(self) -> int:
        return self.q.pop(0)

    def top(self) -> int:
        return self.q[0]

    def empty(self) -> bool:
        return len(self.q) == 0
```

#### [155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

```python
class MinStack:
    def __init__(self):
        self.stk = []
        self.min_stk = [math.inf]

    def push(self, val: int) -> None:
        self.stk.append(val)
        self.min_stk.append(min(val, self.min_stk[-1]))

    def pop(self) -> None:
        self.stk.pop()
        self.min_stk.pop()

    def top(self) -> int:
        return self.stk[-1]

    def getMin(self) -> int:
        return self.min_stk[-1]
```

#### [20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

```
输入：s = "()[]{}"
输出：true
```

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stk = []
        for i in s:
            if i == '(':
                stk.append(')')
            elif i == '{':
                stk.append('}')
            elif i == '[':
                stk.append(']')
            elif stk and i == stk[-1]:
                stk.pop()
            else: 
                return False
        return not len(stk)
```



#### [1190. 反转每对括号间的子串](https://leetcode-cn.com/problems/reverse-substrings-between-each-pair-of-parentheses/)

请你按照从括号内到外的顺序，逐层反转每对匹配括号中的字符串，并返回最终的结果。

```
输入：s = "(abcd)"
输出："dcba"

输入：s = "(u(love)i)"
输出："iloveu"
```

**模拟遍历**：

+ 创建一个放字符串的栈, 以及一个保存当前字符的变量
+ 遇到` ( `就将当前的字符串推入栈, 并将当前字符串其设置为空
+ 遇到` )` 就将当前的字符串反转, 然后与栈的顶部元素合并, 将栈的顶部元素弹出
+ 遇到普通的字符就将其添加到当前字符串的尾部
+ 遍历结束返回字符串

```python
class Solution:
    def reverseParentheses(self, s: str) -> str:
        Str = ''
        stack = []
        for ch in s:
            if ch == '(':
                stack.append(Str)
                Str = ''
            elif ch == ')':
                Str = stack.pop() + ''.join(reversed(Str))
            else:
                Str += ch
        return Str

```

**括号预处理**：

法一需要翻转字符串，复杂度为O(n^2)。但其实只需要得到一个正确的顺序即可。

因此我们可以对括号进行预处理，交换其左右括号的坐标。因此我们在向右遍历到左括号时实则跳到了右括号向左遍历。

https://leetcode-cn.com/problems/reverse-substrings-between-each-pair-of-parentheses/solution/zhan-dong-tu-yan-shi-by-xiaohu9527-hua8/

```python
class Solution:
    def reverseParentheses(self, s: str) -> str:
        n = len(s)
        stk = []
        Next = [0] * n
        for i in range(n):
            if s[i] == '(':
                stk.append(i)
            elif s[i] == ')':
                j = stk.pop()
                Next[i], Next[j] = j, i
        print(Next)  #记录括号翻转后的下标
        ans = ''
        dirc = 1
        i = 0
        while i < n:
            if s[i] == '(' or s[i] == ')':
                i = Next[i]
                dirc = -dirc #改变遍历方向
            else:
                ans += s[i]
            i += dirc
        return ans
```

### 单调栈

#### [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

请根据每日 `气温` 列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。

```
输入：temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
输出: [1, 1, 4, 2, 1, 1, 0, 0]
```

单调栈：在遍历数组时用栈把数组中数的下标存起来，如果当前遍历的数比栈顶元素来的大，说明栈顶元素的下一个比它大的数就是当前元素。

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        length = len(temperatures)
        ans = [0] * length
        stack = []
        for i in range(length):
            temperature = temperatures[i]
            while stack and temperature > temperatures[stack[-1]]:
                prev_index = stack.pop()
                ans[prev_index] = i - prev_index
            stack.append(i)
        return ans
```

#### [503. 下一个更大元素 II](https://leetcode-cn.com/problems/next-greater-element-ii/)

给定一个循环数组（最后一个元素的下一个元素是数组的第一个元素），输出每个元素的下一个更大元素。

```
输入: [1,2,1]
输出: [2,-1,2]
```

遍历2n-1次数组，其他与739题类似，栈依旧保存下标。

```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        stk = []
        n = len(nums)
        nums = nums + nums[0:n-1]
    
        res = [-1] * len(nums)
        for i in range(len(nums)):
            while stk and nums[stk[-1]] < nums[i]:
                index = stk.pop()
                res[index] = nums[i]
            stk.append(i)
        return res[0:n]
```

```js
var nextGreaterElements = function(nums) {
    const n = nums.length
    let res = new Array(n).fill(-1)
    const stk = []
    for(let i = 0; i < n * 2 - 1; i++) {
        while(stk.length && nums[stk[stk.length - 1]] < nums[i % n]) {
            res[stk[stk.length - 1]] = nums[i % n]
            stk.pop()
        }
        stk.push(i % n)
    }
    return res
};
```

#### [84. 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

![1639536773263](D:\leetcode题目整理\leetcode-栈.assets\1639536773263.png)

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stk = []
        res = 0
        heights = [0] + heights + [0]
        for i in range(len(heights)):
            while stk and heights[stk[-1]] > heights[i]:
                index = stk.pop()
                res = max(res, heights[index] * (i - stk[-1] - 1)) #注意宽度是i-stk[-1]-1
            stk.append(i)
        return res
```

#### 84题进阶版：[5655. 重新排列后的最大子矩阵](https://leetcode-cn.com/problems/largest-submatrix-with-rearrangements/)

#### [316. 去除重复字母](https://leetcode-cn.com/problems/remove-duplicate-letters/)

给你一个字符串 `s` ，请你去除字符串中重复的字母，使得每个字母只出现一次。需保证 **返回结果的字典序最小**

```
输入：s = "bcabc"
输出："abc"
```

```js
/**
 * @think 利用栈和贪心算法的思想
 *        1. 维护一个栈stack，对字符串进行正序遍历
 *        2. 对每个字符char，首先判断stack中是否存在，
 *          2.1 若stack栈顶值比char大且后续还存在此值，则将栈顶弹出；
 *            2.1.1 使用indexOf(xx, i)取代 lastIndexOf(xx)减少遍历次数会更快
 *        3. 入栈每个char
 *        4. 打印栈底到栈顶即为结果
 * @time O(nlogn) 
 * @space 0(1) 只需借用一个栈
 * @param {string} s
 * @return {string}
 */
var removeDuplicateLetters = function(s) {
    var stk = []
    for(let i = 0; i < s.length; i++) {
        let char = s[i]
        if(stk.indexOf(char) > -1) continue
        while(stk.length > 0 && stk[stk.length-1] > char && s.indexOf(stk[stk.length-1], i)>i){
            stk.pop()
        }
        stk.push(char)
    }
    return stk.join('')
};
```

#### [402. 移掉 K 位数字](https://leetcode-cn.com/problems/remove-k-digits/)

与316题类似

给定一个以字符串表示的非负整数` num`，移除这个数中的` k `位数字，使得剩下的数字最小。

```
输入: num = "1432219", k = 3
输出: "1219"
解释: 移除掉三个数字 4, 3, 和 2 形成一个新的最小的数字 1219。
```

**解题**：**贪心+单调栈**

我们可以得出「删除一个数字」的贪心策略：

**对于两个数 123a456 和 123b456**，如果 **a > b**， 那么数字 **123a456 大于 数字 123b456，否则数字 123a456 小于等于数字 123b456**。也就说，两个相同位数的数字大小关系取决于第一个不同的数的大小。

基于此，我们可以每次对整个数字序列执行一次这个策略；删去一个字符后，剩下的长度的数字序列就形成了新的子问题，可以继续使用同样的策略，直至删除`k`次。

因此我们的思路就是：

+ 从左到右遍历
+ 对于遍历到的元素，我们选择保留。
+ 但是我们可以选择性丢弃前面相邻的元素。
  丢弃与否的依据如上面的前置知识中阐述中的方法。
+ 每次丢弃一次，k 减去 1。当 k 减到 0 ，我们可以提前终止遍历。
  而当遍历完成，如果 k 仍然大于 0。不妨假设最终还剩下 x 个需要丢弃，那么我们需要选择删除末尾 x 个元素。

```js
var removeKdigits = function (num, k) {
  const stack = [];
  for (let i = 0; i < num.length; i++) {
    const c = num[i];
    while (k > 0 && stack.length && stack[stack.length - 1] > c) {
      stack.pop();
      k--;
    }
    if (c != '0' || stack.length != 0) {
      stack.push(c);
    }
  }
  while (k > 0) {
    stack.pop();
    k--;
  }
  return stack.length == 0 ? "0" : stack.join('');
};
```

#### [321. 拼接最大数](https://leetcode-cn.com/problems/create-maximum-number/)

402题进阶版

给定长度分别为` m `和 `n` 的两个数组，其元素由` 0-9 `构成，表示两个自然数各位上的数字。现在从这两个数组中选出` k (k <= m + n) `个数字拼接成一个新的数，要求从同一个数组中取出的数字保持其在原数组中的相对顺序。

```
输入:
nums1 = [3, 4, 6, 5]
nums2 = [9, 1, 2, 5, 8, 3]
k = 5
输出:
[9, 8, 6, 5, 3]
```

```js
var maxNumber = function(nums1, nums2, k) {
    //4,9,1,1
    const pickMax = (num, k) => {
        let stack = []
        //2
        let drop = num.length - k
        for(let c of num) {
            while(stack && drop && stack[stack.length-1] < c) {
                stack.pop()
                drop--
            }
            stack.push(c)
        }
        console.log(stack.slice(0,k).join(''))
        return stack.slice(0,k).join('')
    }

    const merge = (n1, n2) => {
        let res = []
        while(n1 || n2) {
            if(n1 > n2) {
                res.push(n1[0])
                n1 = n1.substr(1)
            }else {
                res.push(n2[0])
                n2 = n2.substr(1)
            }
        }
        return res.join('')
    }

    let max = ''
    for(let i = 0; i <= k; i++) {
        if (i <= nums1.length && k - i <= nums2.length) {
            let res = merge(pickMax(nums1, i), pickMax(nums2, k-i))
            if(res > max) max = res
        }
    }
    
    return max.split('')
};
```

