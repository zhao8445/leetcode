from typing import List


def merge(intervals: List[List[int]]) -> List[List[int]]:
    """合并区间"""
    intervals.sort(key=lambda x: x[0])

    print(intervals)

    merged = []
    for interval in intervals:
        # 如果列表为空，或者当前区间与上一区间不重合（上一个区间的右端值 < 下一个区间的左端值，直接添加
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # 否则的话，我们就可以与上一区间进行合并
            merged[-1][1] = max(merged[-1][1], interval[1])
            # merged[-1][1] = interval[1]
            """
            输入：
            [[1,4],[2,3]]
            输出：
            [[1,3]]
            预期结果：
            [[1,4]]
            4 > 3 所以合并的区间右端值应该取第一个区间右端值4和第二个区间左端值3中较大的值，
            即max(merged[-1][1], interval[1])
            """

        print(interval, " ", merged, "-1", merged[-1])

    return merged

intervals = [[1,3],[2,6],[8,10],[15,18]]
merge(intervals)
