# leetcode-位运算

#### [461. 汉明距离](https://leetcode-cn.com/problems/hamming-distance/)

两个整数之间的[汉明距离](https://baike.baidu.com/item/汉明距离)指的是这两个数字对应二进制位不同的位置的数目。给出两个整数 `x` 和 `y`，计算它们之间的汉明距离。

```
输入: x = 1, y = 4
输出: 2
解释:
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑
```

对两个数进行异或操作，位级表示不同的那一位为 1，统计有多少个 1 即可。

```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        res = x ^ y
        count = 0
        while res > 0:
            if res & 1 == 1:
            	count += 1
            res = res >> 1
        return count
```

`x & (x - 1)`删去`x`最右侧的1的结果。

```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        res = x ^ y
        count = 0
        while res > 0:
            res = res & (res - 1)
            count += 1
        return count
```

#### [136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

示例 1:

```
输入: [2,2,1]
输出: 1
```

#### 解题

```js
var singleNumber = function(nums) {
    let res = 0
    for(const ch of nums) {
        //两个相同的数异或为0
        res ^= ch
    }
    return res
};
```

#### [268. 丢失的数字](https://leetcode-cn.com/problems/missing-number/)

给定一个包含 `[0, n]` 中 `n` 个数的数组 `nums` ，找出 `[0, n]` 这个范围内没有出现在数组中的那个数。

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        res = len(nums)
        for i, num in enumerate(nums):
            res ^= i ^ num #两个相同的数异或为0
        return res
```



#### [137. 只出现一次的数字 II](https://leetcode-cn.com/problems/single-number-ii/)

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。

示例 1:

```
输入: [2,2,3,2]
输出: 3
```

#### 解题

也可以用`哈希表计数`和`set求和`的方式

**位运算**

```js
var singleNumber = function(nums) {
    let seenOnce = 0, seenTwice = 0;
    for (let num of nums) {
      seenOnce = ~seenTwice & (seenOnce ^ num);
      seenTwice = ~seenOnce & (seenTwice ^ num);
    }
    return seenOnce;
};
```

#### [260. 只出现一次的数字 III](https://leetcode-cn.com/problems/single-number-iii/)

给定一个整数数组` nums`，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。

示例 :

```
输入: [1,2,1,3,2,5]
输出: [3,5]
```

- 异或运算：`x ^ 0 = x` ， `x ^ 1 = ~x`
- 与运算：`x & 0 = 0` ， `x & 1 = x`

```js
// [1,2,1,3,2,5]
// xor = 3 ^ 5 = 001
// mask = 001 & 111 = 001
// 1^001=1 2^001=0 3^001=1 5^001=0
//a & (-a) 可以获得最低的非0位，最低的1位就表示这个位他们一定不同，这样我们就可以根据这个位进行分组。
var singleNumber = function(nums) {
  let xor = 0
  for (let i = 0; i < nums.length; i++) {
    xor ^= nums[i]
  }
  let mask = xor & (-xor)
  let ans = Array(2)
  for (let i = 0; i < nums.length; i++) {
    if ((nums[i] & mask) === 0) {
      ans[0] ^= nums[i]
    } else {
      ans[1] ^= nums[i]
    }
  }
  return ans
}
```

#### [477. 汉明距离总和](https://leetcode-cn.com/problems/total-hamming-distance/)

两个整数的 [汉明距离](https://baike.baidu.com/item/汉明距离/475174?fr=aladdin) 指的是这两个数字的二进制数对应位不同的数量。计算一个数组中，任意两个数之间汉明距离的总和。

```
输入: 4, 14, 2
输出: 6
```

逐位统计

只需要统计出所有元素二进制第i位共有c个1，n-c个0，则这一位上的汉明距离为c*(n-c)，将所有位数上的距离累加，可以用(val >> i) & 1取出第i位的值。

```python
class Solution:
    def totalHammingDistance(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(30):
            c = sum(((val >> i) & 1) for val in nums)
            ans += c * (n - c)
        return ans
```

