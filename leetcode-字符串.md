# leetcode-字符串

[TOC]

#### 1. 字符串循环移位包含

编程之美 3.1

```
s1 = AABCD, s2 = CDAA
Return : true
```

给定两个字符串 s1 和 s2，要求判定 s2 是否能够被 s1 做循环移位得到的字符串包含。

s1 进行循环移位的结果是 s1s1 的子字符串，因此只要判断 s2 是否是 s1s1 的子字符串即可。

#### 2. 字符串循环移位

编程之美 2.17

```
s = "abcd123" k = 3
Return "123abcd"
```

将字符串向右循环移动 k 位。

将 abcd123 中的 abcd 和 123 单独翻转，得到 dcba321，然后对整个字符串进行翻转，得到 123abcd。

#### 3. 字符串中单词的翻转

程序员代码面试指南

```
s = "I am a student"   //I ma a tenduts
Return "student a am I"
```

将每个单词翻转，然后将整个字符串翻转。

#### 4. 两个字符串包含的字符是否完全相同

[力扣242](https://leetcode-cn.com/problems/valid-anagram/)

```
s = "anagram", t = "nagaram", return true.
s = "rat", t = "car", return false.
```

可以用 HashMap 来映射字符与出现次数，然后比较两个字符串出现的字符数量是否相同。

由于本题的字符串只包含 26 个小写字符，因此可以使用长度为 26 的整型数组对字符串出现的字符进行统计，不再使用 HashMap。

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        Map = [0] * 26
        for i in s:
            Map[ord(i) - 97] += 1
        for j in t:
            Map[ord(j) - 97] -= 1
        for m in Map:
            if m != 0:
                return False
        return True
```

#### 5. 计算一组字符集合可以组成的回文字符串的最大长度

[力扣409](https://leetcode-cn.com/problems/longest-palindrome/)

```
输入:"abccccdd"
输出:7
解释:我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。
```

```python
class Solution:
    def longestPalindrome(self, s: str) -> int:
        Map = [0] * 58
        for i in s:
            Map[ord(i) - 65] += 1
        res = 0
        for m in Map:
            res += m // 2 * 2
        if res < len(s):
            res += 1 
        return res
```

#### 6. 字符串同构

[力扣205](https://leetcode-cn.com/problems/isomorphic-strings/)

```
输入：s = "egg", t = "add"  输出：true
输入：s = "foo", t = "bar"  输出：false
```

index对比

记录一个字符上次出现的位置，如果两个字符串中的字符上次出现的位置一样，那么就属于同构。

```python
class Solution:
    def isIsomorphic(self, s, t):
        for i in range(len(s)):
            if s.index(s[i]) != t.index(t[i]):
                return False
        return True
```

hash表 双向记录

```python
class Solution:
    def isIsomorphic(self, s, t):
        x = {}
        y = {}
        for i in range(len(s)):
            if (s[i] in x and x[s[i]] != t[i]) or (
                    t[i] in y and y[t[i]] != s[i]):
                return False
            x[s[i]] = t[i]
            y[t[i]] = s[i]
        return True
```

#### 7. 回文子字符串个数

[力扣647](https://leetcode-cn.com/problems/palindromic-substrings/)

```
输入："abc"
输出：3
解释：三个回文子串: "a", "b", "c"

输入："aaa"
输出：6
解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
```

中心点遍历：回文串的中心点可能是1个字符可能是2个字符，从前往后依次遍历每一个每两位为中心点的子串，尝试去扩展。O(n^2)

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        res = 0
        for i in range(n):
            for j in range(0, 2):
                l = i
                r = i + j
                while l >= 0 and r < n and s[l] == s[r]:
                    l -= 1
                    r += 1
                    res += 1
        return res
```

动态规划解法O(n^2)

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        res = 0
        for j in range(n):
            for i in range(j+1): #遍历方向很重要
                if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                    res += 1
        return res
```

#### 8. 判断一个整数是否是回文数

[力扣9](https://leetcode-cn.com/problems/palindrome-number/)

不能将整数转为字符串

将整数分成左右两部分，右边那部分需要转置，然后判断这两部分是否相等。

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x == 0:
            return True
        if x < 0 or x % 10 == 0:
            return False
        r = 0
        while x > r:
            r = r * 10 + x % 10
            x //= 10
        return r == x or r // 10 == x
```

#### 9. 统计二进制字符串中连续 1 和连续 0 数量相同的子字符串个数

[力扣696](https://leetcode-cn.com/problems/count-binary-substrings/)

```
输入：s = "00110011"
输出：6
解释：6 个子串满足具有相同数量的连续 1 和 0 ："0011"、"01"、"1100"、"10"、"0011" 和 "01" 。
注意，一些重复出现的子串（不同位置）要统计它们出现的次数。
另外，"00110011" 不是有效的子串，因为所有的 0（还有 1 ）没有组合在一起。
```

将字符串 s 按照 0 和1的连续段分组，存在`counts`数组中。例如 s = 00111011，可以得到counts={2,3,1,2}。

两个相邻的数一定代表的是两种不同的字符。只要遍历所有相邻的数对，求它们的贡献总和，即可得到答案。

复杂度O(n)

```python
class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        n = len(s)
        counts = []
        index = 0
        while index < n:
            c = s[index]
            count = 0
            while index < n and s[index] == c:
                count += 1
                index += 1
            counts.append(count)
        res = 0
        for i in range(1, len(counts)):
            res += min(counts[i], counts[i-1])
        return res
```

更精简的写法：用两个变量存储相邻两种不同字符的长度

```python
class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        n = len(s)
        prelen = 0
        curlen = 1
        res = 0
        for i in range(1, n):
            if s[i] == s[i-1]:
                curlen += 1
            else:
                prelen = curlen
                curlen = 1
            
            if prelen >= curlen:
                res += 1
        return res
```

