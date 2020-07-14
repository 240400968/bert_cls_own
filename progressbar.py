# _*_ coding:utf-8 _*_

"""
#=============================================================================
#  ProjectName: scripts
#     FileName: core
#         Desc: 闲来无事写两个进度条样式
#       Author: Jeeyshe
#        Email: jeeyshe@gmail.com
#     HomePage: lujianin.com
#       Create: 2019/11/25 下午2:21
#=============================================================================
"""

from __future__ import division
import sys

PY2 = sys.version_info[0] == 2


def charBar(action, total, current, fill_with="#", point="=>"):
    """
    安装依赖########################=>100%
    action: 任务描述
    total: 总进度
    current: 当前进度
    fill_with: 以什么字符填充
    point: 以什么字符指向
    """
    progress = int((current / total) * 100)
    sys.stdout.write('\r' + action + fill_with * progress + point + str(progress) + '% ')
    sys.stdout.flush()


def viewBar(action, total, current):
    """
    安装依赖 ----->: 48%
    action: 任务描述
    total: 总量
    current: 当前进度
    """
    output = sys.stdout
    output.write('\r %s ----->: %.0f%% ' % (action, (current / total) * 100))
    output.flush()


class ShowProcess(object):
    max_arrow = 100  # 进度条总长度

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_step, started='-', worked='>', done='Done'):
        self.max_step = max_step
        self.worked = worked
        self.started = started
        self.i = 0
        self.infoDone = done
        # 显示函数，根据当前的处理进度i显示进度
        # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100%

    def show_process(self, i=None):
        self.i = i if i else self.i + 1
        num_arrow = int(self.i * self.max_arrow / self.max_step)  # 计算显示多少个'>'
        num_line = self.max_arrow - num_arrow  # 计算显示多少个'-'
        percent = self.i * 100 / self.max_step  # 计算完成进度，格式为xx.xx%
        progress_bar = '[' + self.worked * num_arrow + self.started * num_line + ']' + '%.0f' % percent + '%' + '\r'
        sys.stdout.write(progress_bar)  # 这两句打印字符到终端
        sys.stdout.flush()
        if self.i >= self.max_step:
            self.close()

    def close(self):
        print("")
        print(self.infoDone)
        self.i = 0


if __name__ == '__main__':
    import time

    # for i in range(1, 101):
    #     viewBar("安装依赖", 100, i)
    #     time.sleep(0.1)
    # print()

    max_steps = 100
    process_bar = ShowProcess(max_steps, '')
    for i in range(max_steps):
        process_bar.show_process()
        time.sleep(0.1)
