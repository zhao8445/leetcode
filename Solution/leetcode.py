import time
from typing import List


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:

    # 二叉树深度
    def maxDepth(self, root):
        deque = [root]
        res = 0
        if not root:
            return 0
        while deque:
            tmp = []
            for node in deque:
                if node.left:
                    tmp.append(node.left)
                if node.right:
                    tmp.append(node.right)
            deque = tmp
            res += 1
        return res

    # 链表中倒数第K个节点
    def getKthFromEnd(self, head, k):
        fast, low = head, head
        for _ in range(k):
            fast = fast.next
        while fast:
            fast = fast.next
            low = low.next
        return low

    # 二叉树的镜像
    def mirrorTree(self, root: TreeNode):
        if not root:
            return
        deque = [root]
        while deque:
            tmp = []
            for node in deque:
                if node.left:
                    tmp.append(node.left)
                if node.right:
                    tmp.append(node.right)
                node.left, node.right = node.right, node.left
            deque = tmp
        return root

    # 两数之和
    def twoSum(self, nums, target):
        hashMap = {}
        for i in range(len(nums)):
            hashMap[nums[i]] = i
        for i in range(len(nums)):
            k = hashMap.get(target - nums[i])
            if k is not None and k != i:
                return [i, k]

    # 0901 两数相加
    def addTwoNumbers(self, l1, l2):
        p1, p2  = l1, l2
        cur = ListNode(0)
        dum = cur
        carry = 0

        while p1 or p2:
            x = p1.val if p1 else 0 # 如果p1为空则x为0，否则p1.val
            y = p2.val if p2 else 0
            sum = (x + y + carry) % 10
            carry = (x + y + carry) // 10
            if cur:
                # cur.val = sum
                # cur.next = ListNode(0) # 未初始化next节点，导致cur.next报错NoneType
                cur.next = ListNode(sum) # 初始化头结点为0，从头结点next节点才开始存入计算的值
                cur = cur.next
            if p1:
                p1 = p1.next
            if p2:
                p2 = p2.next
            if carry > 0:
                cur.next = ListNode(carry)
        return dum.next

    # 0901 无重复字符的最长子串
    def lengthOfLongestSubstring(self, s: str) -> int:
        hashMap = {}
        i, j = 0, 0
        length = len(s)
        count = 0

        if not s:
            return 0

        while i < length and j < length:
            if s[j] in hashMap:
                # count = max(count, j - i) // 写在这个位置，会导致无重复字符串如dwsre，不进入条件计算count
                # i = j
                '''
                abcabcbb 0 0  0 {'a': 0}
                abcabcbb 0 1 a 1 {'a': 0, 'b': 1}
                abcabcbb 0 2 ab 2 {'a': 0, 'b': 1, 'c': 2}
                abcabcbb 3 3  2 {'a': 3, 'b': 1, 'c': 2}
                abcabcbb 4 4  2 {'a': 3, 'b': 4, 'c': 2}
                abcabcbb 5 5  2 {'a': 3, 'b': 4, 'c': 5}
                abcabcbb 6 6  2 {'a': 3, 'b': 6, 'c': 5}
                abcabcbb 7 7  2 {'a': 3, 'b': 7, 'c': 5}
                '''
                i = max(hashMap.get(s[j]), i) # "abba"
                # i = hashMap.get(s[j])
            # hashMap[s[j]] = j
            """
            input: " "
            output: 0
            expect: 1
            """
            hashMap[s[j]] = j + 1
            # count = max(count, j - i)
            count = max(count, j - i + 1)
            print(s, s[j], i, j, s[i:j+1], count, hashMap)
            j += 1
        return count

    def subsets(self, nums):
        """ 9/18 78.子集 """
        output = [[]]

        for num in nums:
            # output += [curr + [num] for curr in output]
            # tmp = [curr + [num] for curr in output]
            tmp = []
            print("#########", output)
            for curr in output:
                tmp_in = curr + [num]
                tmp += [tmp_in]
                print(curr, "+", [num], "=", tmp_in, "+=", tmp)
            output += tmp
            # print(tmp)
        print("#########", output)
        return output

    def singleNumber(self, nums):
        """9/23 数组中数字出现的次数"""
        hashMap = {}
        for num in nums:
            if num in hashMap:
                hashMap[num] += 1
            else:
                hashMap[num] = 1
        for k,v in hashMap.items():
            if v == 1:
                return k

    def merge(self, A: List[int], m: int, B: List[int], n: int) -> None:
        """9/23 合并两个有序数组"""
        res = []
        A[:] = A[:m] # [1,0],[2] -> [1,2,0]
        B[:] = B[:n]
        i, j = 0, 0

        while i < m and j < n:
            if A[i] < B[j]:
                res.append(A[i])
                i += 1
            else:
                res.append(B[j])
                j += 1
        if i < m:
            res[i + j:] = A[i:]
        if j < n:
            res[i + j:] = B[j:]
        A[:] = res

    def isSymmetric(self, root: TreeNode) -> bool:
        """9/28 对称的二叉树"""
        def dfs(L, R):
            if not L and not R:
                return True
            if not L or not R or L.val != R.val:
                return False
            return dfs(L.right, R.left) and dfs(L.left, R.right)
        return dfs(root.left, root.right) if root else True

    def removeDuplicates(self, nums: List[int]) -> int:
        """9/28 删除排序数组中的重复项"""
        p, q = 0, 1
        n = len(nums)

        while q < n:
            if nums[p] != nums[q]:
                p += 1
                nums[p] = nums[q]
            q += 1
        return p + 1

    def getNumberOfK(self, data, k):
        """9/29 排序数组中查找数字"""
        l, r =0, len(data)
        while l < r:

            mid = (l + r) // 2
            print(mid, l, r)
            if data[mid] < k:
                l = mid + 1
            else:
                r = mid

    def hasCycle(self, head: ListNode) -> bool:
        """9/30 环形链表"""
        hashMap = {}
        while head:
            if head not in hashMap:
                hashMap[head] = True
            else:
                return True
            head = head.next
        return False

    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        '''19.删除链表的倒数第N个节点'''
        # fast, slow = head, head
        '''
        faile:[1] 快慢指针初始指向null节点dummy，dummy下一节点指向head
        '''
        dummy = ListNode(0)
        dummy.next = head

        slow, fast = dummy, dummy

        i = 0
        while i < n:
            fast = fast.next
            i += 1
        # while fast:
        '''
        输入
        [1,2,3,4,5]
        2
        输出
        [1,2,3,4]
        预期结果
        [1,2,3,5]
        最终会导致fast指向None，slow指向4，删掉的是5节点
        '''
        while fast and fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next

    def singleNumber(self, nums: List[int]) -> int:
        '''136.只出现一次的数字'''
        hashMap = {}
        for num in nums:
            if num in hashMap:
                hashMap[num] += 1
            else:
                hashMap[num] = 1
        for key, value in hashMap.items():
            if value == 1:
                return key

    def longestPalindrome(self, s: str) -> str:
        """5.最长回文字串"""
        size = len(s)
        if size < 2:
            return s
        dp = [[False for _ in range(size)] for _ in range(size)]
        max_len = 1
        start = 0
        for i in range(size):
            dp[i][i] = True

        for j in range(1, size):
            for i in range(0, j):
        # for i in range(0, size):
        #     for j in range(i+1, size):
                """
                input: s = "aaaa" 
                output: aaa
                expacted: aaaa
                
                0 1 aa dp[ 0 ][ 1 ]= True
                0 2 aaa dp[ 0 ][ 2 ]= True
                0 3 aaa dp[ 0 ][ 3 ]= False
                1 2 aaa dp[ 1 ][ 2 ]= True
                1 3 aaa dp[ 1 ][ 3 ]= True
                2 3 aaa dp[ 2 ][ 3 ]= True
                
                dp[0][3] = s[0] == s[3] and dp[0 + 1][3 - 1]
                因为未先对dp[1][2]的值进行判断，为false，所以dp[0][3]为false，实际dp[0][3]应该为true
                体现在dp[i][j]的二维数组上，就是每一列一列竖着遍历，01/02 12/03 13 23/04 14 24 34的顺序
                为何要竖着遍历？即计算dp[i][j]时，先判断dp[i+1][j-1],即dp[i][j]二维数组，先判断左下角
                """
                if s[i] == s[j]:
                    if j - i < 3:  # 即j - i + 1 < 4 就是i j之间的长度为2个或3个字符
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                else:
                    dp[i][j] = False

                if dp[i][j]:
                    cur_len = j - i + 1
                    if cur_len > max_len:
                        max_len = cur_len
                        start = i
                    max_len = max(max_len, cur_len)
                    start = i
                    print(start,max_len, i, j, s[start:start + max_len], "dp[",i,"][",j,"]=",dp[i][j])
        print(s[start:start + max_len])
        return s[start:start + max_len]

    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        '''25.K个一组翻转链表'''
        res = []
        ans = []
        count = k

        dum = ListNode(0)
        cur = dum

        dummy = ListNode(-1)
        dummy.next = head
        """
        while head:
            if count > 0:
                res.append(head)
                head = head.next
                count -= 1
                print(count)
                
        input:[1,2] k=2 
        output:[1,2]
        expact:[2,1]
        
        当k正好为链表长度时，不会进入出栈循环，因为此时正好结束链表遍历了。
        创建一个新链表节点dummy，dummy.next = head，遍历dummy，在赋值时dummy.next，即可解决
        """
        while dummy:
            if count > 0:
                res.append(dummy.next)
                dummy = dummy.next
                count -= 1
            else:
                while count < k:
                    node = res.pop()
                    """
                    cur = node
                    cur = cur.next
                    此时cur, dummy, head为同一内存地址，操作cur，dummy和head的指向下一节点的指针方向也会改变
                    如原链表dummy：1 -> 2 -> 3 -> 4 -> 5
                    再操作过后： 1 -> 2 -> 1 -> 2 -> 1
                    """
                    count += 1
                    ans.append(node)
        while res:
            """
            例如[1,2,3,4,5],k=2，两个一翻转后，还剩5未赋值，暂存到res中，故拼接到后面，否则输出[1,2,3,4]
            """
            ans.append(res.pop(0))
            """
            因为剩下小于k个节点时，要保留原顺序，故先进先出，保持进栈顺序
            """
            # cur.next = res.pop(0);
            # cur = cur.next
        for ans_node in ans:
            cur.next = ans_node
            cur = cur.next
            # res.append(ans_node.val)
        while dum:
            if dum.next:
                res.append(dum.next.val)
            dum = dum.next
        print(res)
        return dum

    def lengthOfLIS(self, nums: List[int]) -> int:
        """最长上升子序列"""
        if not nums: return 0
        dp = [1] * len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:  # 如果要求非严格递增，将此行 '<' 改为 '<=' 即可。
                    dp[i] = max(dp[i], dp[j] + 1)
                    # dp[i] = dp[j] + 1
                """
                dp[i]记录的是从nums[0]开始遍历, 小于nums[i]的递增序列数字个数
                从0开始，依次与nums[i]比较
                [4, 10, 5, 6]     [3, 6, 4, 4]
                小于4的递增序列数字个数为3个，6>4，那么6的递增子序列个数为4的递增子序列个数+1,3+1，即dp[j]+1
                
                为何要取max(dp[i], dp[j] + 1)
                eg:
                [1, 3, 6, 7, 9, 4, 10]     [1, 2, 3, 4, 5, 3, 2]
                [3, 6, 7, 9, 4, 10]     [2, 3, 4, 5, 3, 3]
                [6, 7, 9, 4, 10]     [3, 4, 5, 3, 4]
                [7, 9, 4, 10]     [4, 5, 3, 5]
                [9, 4, 10]     [5, 3, 6]
                当10与9比较时，递增子序列为[1 3 6 7 9 10]，个数为6个
                [4, 10]     [3, 4]
                
                当10与4比较时，递增子序列为[1 3 4 10]，个数为4个，覆盖了上一次数据6个，显然不是预期结果
                故每一次取dp[i]原数值和dp[j]+1的最大值 
                """
                print(nums[j:i+1],"   ", dp[j:i+1])

            print("__________")
        return max(dp)


if __name__ == "__main__":
    so = Solution()
    node1 = ListNode(1)
    node2 = ListNode(2)
    node3 = ListNode(3)
    node4 = ListNode(4)
    node5 = ListNode(5)
    node1.next = node2
    # node2.next = node3
    # node3.next = node4
    # node4.next = node5
    # s.reverseKGroup(node1, 2)
    nums = [1,3,6,7,9,4,10,5,6]
    so.lengthOfLIS(nums)











