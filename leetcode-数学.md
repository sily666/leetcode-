# leetcode-数学

### 1.素数

#### [204. 计数质数](https://leetcode-cn.com/problems/count-primes/)

统计所有小于非负整数 *`n`* 的质数的数量。

```
输入：n = 10
输出：4
解释：小于 10 的质数一共有 4 个, 它们是 2, 3, 5, 7 。
```

埃拉托斯特尼筛法在每次找到一个素数时，将能被素数整除的数排除掉。

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        count = 0
        isPrime = [1] * n
        for i in range(2, n):
            if isPrime[i]:
                count += 1
                for j in range(i * i, n, i):
                    isPrime[j] = 0
        return count
```

### 2.最大公约数

```python
def gcd(a, b):
    return b == 0 ? a : gcd(b, a % b) 
```

最小公倍数为两数的乘积除以最大公约数。

```python
def lcm(a, b):
    return a * b / gcd(a, b)
```

### 3.进制转换

#### [504. 七进制数](https://leetcode-cn.com/problems/base-7/)

```python
class Solution:
    def convertToBase7(self, num: int) -> str:
        res = ''
        num2 = abs(num)
        while num2:
            res += str(num2 % 7)
            num2 = num2 // 7
        if num > 0: res = res[::-1]
        elif num == 0: res = '0'
        else: res = '-' + res[::-1]
        return res
```

#### [405. 数字转换为十六进制数](https://leetcode-cn.com/problems/convert-a-number-to-hexadecimal/)

给定一个整数，编写一个算法将这个数转换为十六进制数。对于负整数，我们通常使用 [补码运算](https://baike.baidu.com/item/补码/6854613?fr=aladdin) 方法。

**原码和补码**

https://leetcode-cn.com/problems/convert-a-number-to-hexadecimal/solution/python3-si-lu-qing-xi-jian-dan-yi-dong-j-563n/

```python
class Solution:
    def toHex(self, num: int) -> str:
        res = ''
        if num < 0:
            num =(abs(num) ^ (2**32-1)) + 1
        elif num == 0:
            return '0'
        while num:
            a = num % 16
            if a > 9:
                a = chr(a+87)
            else:
                a = str(a)
            res += a 
            num >>= 4
        return res[::-1]
```

#### [168. Excel表列名称](https://leetcode-cn.com/problems/excel-sheet-column-title/)

给定一个正整数，转换为26进制

```python
class Solution:
    def convertToTitle(self, n: int) -> str:
        s = ''
        while n:
            n -= 1
            #ASCII码转大写字符 并且左加 
            s = chr(65 + n % 26) + s
            n //= 26
        return s
```

### 4.阶乘

#### [172. 阶乘后的零](https://leetcode-cn.com/problems/factorial-trailing-zeroes/)

给定一个整数 *n*，返回 *n*! 结果尾数中零的数量。

```
输入: 5
输出: 1
解释: 5! = 120, 尾数中有 1 个零.
```

尾部的 0 由 2 * 5 得来，2 的数量明显多于 5 的数量，因此只要统计有多少个 5 即可。

对于一个数 N，它所包含 5 的个数为：N/5 + N/5^2 + N/5^3 + ...，其中 N/5 表示不大于 N 的数中 5 的倍数贡献一个 5，N/52 表示不大于 N 的数中 52 的倍数再贡献一个 5 ...。

```python
class Solution:
    def trailingZeroes(self, n: int) -> int:
        count = 0
        while n > 0:
            count += n // 5
            n = n // 5
        return count
```

### 5.字符串加法减法

#### [67. 二进制求和](https://leetcode-cn.com/problems/add-binary/)

给你两个二进制字符串，返回它们的和（用二进制表示）。

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        i, j = len(a) - 1, len(b) - 1
        carry = 0
        res = ''
        while carry == 1 or i >= 0 or j >= 0:
            if i >= 0 and a[i] == '1':
                carry += 1
            if j >= 0 and b[j] == '1':
                carry += 1
            res = str(carry % 2) + res
            carry //= 2
            i -= 1
            j -= 1
        return res
```

#### [415. 字符串相加](https://leetcode-cn.com/problems/add-strings/)

给定两个字符串形式的非负整数 `num1` 和`num2` ，计算它们的和。

```python
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        res = ""
        i, j, carry = len(num1) - 1, len(num2) - 1, 0
        while i >= 0 or j >= 0:
            n1 = int(num1[i]) if i >= 0 else 0
            n2 = int(num2[j]) if j >= 0 else 0
            tmp = n1 + n2 + carry
            carry = tmp // 10
            res = str(tmp % 10) + res
            i, j = i - 1, j - 1
        return "1" + res if carry else res
```

### 6.相遇问题

#### [462. 最少移动次数使数组元素相等 II](https://leetcode-cn.com/problems/minimum-moves-to-equal-array-elements-ii/)

```
Input:
[1,2,3]

Output:
2

Explanation:
Only two moves are needed (remember each move increments or decrements one element):

[1,2,3]  =>  [2,2,3]  =>  [2,2,2]
```

每次可以对一个数组元素加一或者减一，求最小的改变次数。

```python
class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        nums.sort()
        i, j = 0, len(nums) - 1
        count = 0
        while i < j:
            count += nums[j] - nums[i]
            i += 1
            j -= 1
        return count
```

### 7.多数投票问题

#### [169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

给定一个大小为 *n* 的数组，找到其中的多数元素。多数元素是指在数组中出现次数 **大于** `⌊ n/2 ⌋` 的元素。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        nums.sort()
        return nums[len(nums) // 2]
```

可以利用 Boyer-Moore Majority Vote Algorithm 来解决这个问题，使得时间复杂度为 O(N)。可以这么理解该算法：使用 cnt 来统计一个元素出现的次数，当遍历到的元素和统计元素不相等时，令 cnt--。如果前面查找了 i 个元素，且 cnt == 0，说明前 i 个元素没有 majority，或者有 majority，但是出现的次数少于 i / 2，因为如果多于 i / 2 的话 cnt 就一定不会为 0。此时剩下的 n - i 个元素中，majority 的数目依然多于 (n - i) / 2，因此继续查找就能找出 majority。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        cnt = 0
        majority = nums[0]
        for num in nums:
            if cnt == 0: majority = num
            if majority == num:
                cnt += 1
            else: cnt -= 1
        return majority
```

### 8.其他

#### [367. 有效的完全平方数](https://leetcode-cn.com/problems/valid-perfect-square/)

给定一个 **正整数** `num` ，编写一个函数，如果 `num` 是一个完全平方数，则返回 `true` ，否则返回 `false` 。

间隔为等差数列，使用这个特性可以得到从 1 开始的平方序列。

```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        subnum = 1
        while num > 0:
            num -= subnum
            subnum += 2
        return num == 0
```

#### [326. 3的幂](https://leetcode-cn.com/problems/power-of-three/)

给定一个整数，写一个函数来判断它是否是 3 的幂次方。如果是，返回 `true` ；否则，返回 `false` 。

```python
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        if n < 1: return False
        while (n % 3) == 0:
            n //= 3
        return n == 1
```

```python
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        return True if n>0 and 1162261467 % n == 0 else False

```

#### [238. 除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)

给定一个数组，创建一个新数组，新数组的每个元素为原始数组中除了该位置上的元素之外所有元素的乘积。

要求时间复杂度为 O(N)，并且不能使用除法。

乘积 = 当前数左边的乘积 * 当前数右边的乘积

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [1] * n
        k = 1
        for i in range(n): #当前数左边的乘积
            res[i] = k
            k *= nums[i]
        k = 1
        for i in range(n-1, -1, -1):
            res[i] *= k  # 左 * 右
            k *= nums[i] # k记录右边的乘积
        return res
```

#### [628. 三个数的最大乘积](https://leetcode-cn.com/problems/maximum-product-of-three-numbers/)

给你一个整型数组 `nums` ，在数组中找出由三个数组成的最大乘积，并输出这个乘积。

方法1：排序 O(NlogN)

```js
var maximumProduct = function(nums) {
    nums.sort((a, b) => a - b);
    const n = nums.length;
    return Math.max(nums[0] * nums[1] * nums[n - 1], nums[n - 1] * nums[n - 2] * nums[n - 3]);
};
```

方法2：线性扫描 O(N)

```python
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        max1 = max2 = max3 = float('-inf')
        min1 = min2 = float('inf')
        for i, num in enumerate(nums):
            if num > max1:
                max3 = max2
                max2 = max1
                max1 = num
            elif num > max2:
                max3 = max2
                max2 = num
            elif num > max3:
                max3 = num
            if num < min1:
                min2 = min1
                min1 = num
            elif num < min2:
                min2 = num
        return max(max1 * max2 * max3, max1 * min1 * min2)
```

